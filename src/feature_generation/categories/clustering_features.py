"""
Clustering-Specific Feature Generator

This module provides feature generators specifically designed for clustering algorithms
(HDBSCAN, K-means, etc.) that should NEVER be used during live trading.

These features are optimized for:
- Better cluster separation
- Improved clustering stability
- Enhanced regime discovery
- Clustering algorithm performance

Key Features:
- Distance-based features for clustering
- Cluster stability metrics
- Regime separation features
- Clustering-specific statistical measures
- Dimensionality reduction features
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
from functools import lru_cache

# Third-party imports
import numpy as np

# Import tprint utilities for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_structured,
    tprint_data_preview, tprint_data_format, tprint_feature_counts,
    tprint_timer, tprint_logged
)
import pandas as pd

# Optional third-party imports
try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE, Isomap
    from sklearn.preprocessing import StandardScaler
    SCIPY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    stats = None
    pdist = None
    squareform = None
    linkage = None
    dendrogram = None
    KMeans = None
    DBSCAN = None
    silhouette_score = None
    calinski_harabasz_score = None
    PCA = None
    FastICA = None
    TSNE = None
    Isomap = None
    StandardScaler = None

# VectorBT optimization imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager as MLUnifiedVectorizationManager
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None
    UnifiedVectorizationManager = None
    MLUnifiedVectorizationManager = None

try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Local imports
from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)


@dataclass
class ClusteringFeatureConfig:
    """Configuration for clustering-specific feature generation."""
    
    # Clustering feature categories to include
    include_distance_features: bool = True
    include_separation_features: bool = True
    include_stability_features: bool = True
    include_dimensionality_features: bool = True
    include_hierarchical_features: bool = True
    
    # Clustering algorithm specific features
    include_hdbscan_features: bool = True
    include_kmeans_features: bool = True
    include_dbscan_features: bool = True
    
    # Feature quality filters (relaxed for clustering)
    min_cluster_separation: Optional[float] = None
    max_feature_correlation: Optional[float] = None
    min_clustering_stability: Optional[float] = None
    
    # Performance optimizations
    enable_parallel_processing: bool = True
    enable_matrix_optimization: bool = True
    max_parallel_workers: int = 4
    
    # Clustering-specific parameters
    n_clusters_range: Tuple[int, int] = (2, 10)
    min_cluster_size: int = 5
    max_cluster_size: int = 100
    
    # Feature selection
    max_features_per_category: int = 50
    total_max_features: int = 200
    enable_feature_selection: bool = False
    
    # Clustering quality weights
    separation_weight: float = 0.4
    stability_weight: float = 0.3
    compactness_weight: float = 0.3
    
    def __post_init__(self) -> None:
        """Set default values based on clustering requirements."""
        if self.min_cluster_separation is None:
            self.min_cluster_separation = 0.1
        if self.max_feature_correlation is None:
            self.max_feature_correlation = 0.8
        if self.min_clustering_stability is None:
            self.min_clustering_stability = 0.2


class ClusteringDistanceGenerator(VectorizedFeatureGenerator):
    """
    Clustering Distance Feature Generator.
    
    Generates distance-based features optimized for clustering algorithms.
    These features help clustering algorithms identify natural groupings.
    """
    
    def __init__(self):
        base_config = FeatureConfig(
            name="clustering_distance",
            category=FeatureCategory.REGIME,
            description="Distance-based features for clustering",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=50,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(base_config, enable_matrix_ops=True)
        
        # Initialize VectorBT optimizers
        # vectorbt_optimizer inherited from base class
        self.unified_optimizer = None
        
        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT rolling optimizer
                # vectorbt_rolling_optimizer inherited from base class
                
                # Initialize unified vectorization manager
                unified_config = {
                    "enable_vectorbt": True,
                    "enable_matrix_ops": True,
                    "enable_gpu": False,
                    "optimization_level": "high"
                }
                self.unified_optimizer = UnifiedVectorizationManager(unified_config)
                
                print("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for ClusteringDistanceGenerator")
            except Exception as e:
                print(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate clustering distance features."""
        features = self.generate_features(data)
        
        if features:
            # Return a composite distance score
            distance_score = np.mean(list(features.values()), axis=0)
            return pd.Series(distance_score, index=data.index, name=self.config.name)
        else:
            return pd.Series(0.5, index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive distance-based clustering features."""
        tprint_debug("🔍 Generating clustering distance features")
        tprint_data_preview(data, "Input Data for Distance Features", max_rows=2, max_cols=5)
        
        features = {}
        
        if len(data) < self.config.min_lookback:
            tprint_warning(f"⚠️ Insufficient data: {len(data)} < {self.config.min_lookback}")
            return features
        
        with tprint_timer("Distance Feature Generation", level="PERFORMANCE"):
            # Price distance features
            tprint_debug("📊 Calculating price distance features")
            price_features = self._generate_price_distance_features(data)
            features.update(price_features)
            tprint_debug(f"✅ Generated {len(price_features)} price distance features")
            
            # Volume distance features
            tprint_debug("📊 Calculating volume distance features")
            volume_features = self._generate_volume_distance_features(data)
            features.update(volume_features)
            tprint_debug(f"✅ Generated {len(volume_features)} volume distance features")
            
            # Volatility distance features
            tprint_debug("📊 Calculating volatility distance features")
            volatility_features = self._generate_volatility_distance_features(data)
            features.update(volatility_features)
            tprint_debug(f"✅ Generated {len(volatility_features)} volatility distance features")
            
            # Cross-feature distance features
            tprint_debug("📊 Calculating cross-feature distance features")
            cross_features = self._generate_cross_feature_distance_features(data)
            features.update(cross_features)
            tprint_debug(f"✅ Generated {len(cross_features)} cross-feature distance features")
        
        tprint_success(f"🎉 Generated {len(features)} total clustering distance features")
        return features

    def _generate_price_distance_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate price-based distance features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Rolling price distances
                price_distances = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    # Calculate pairwise distances within window
                    if len(window_prices) > 1:
                        distances = np.abs(np.diff(window_prices))
                        avg_distance = np.mean(distances)
                        max_distance = np.max(distances)
                        min_distance = np.min(distances)
                        std_distance = np.std(distances)
                        
                        price_distances.extend([
                            avg_distance, max_distance, min_distance, std_distance
                        ])
                    else:
                        price_distances.extend([0, 0, 0, 0])
                
                # Pad with zeros for initial values
                padded_distances = [0] * window + price_distances
                
                features[f'price_avg_distance_{window}'] = np.array(padded_distances[:len(prices)])
                features[f'price_max_distance_{window}'] = np.array(padded_distances[:len(prices)])
                features[f'price_min_distance_{window}'] = np.array(padded_distances[:len(prices)])
                features[f'price_std_distance_{window}'] = np.array(padded_distances[:len(prices)])
                
                # Price cluster density
                price_density = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate density as inverse of average distance
                        avg_dist = np.mean(np.abs(np.diff(window_prices)))
                        density = 1 / (avg_dist + 1e-8)
                        price_density.append(density)
                    else:
                        price_density.append(0)
                
                padded_density = [0] * window + price_density
                features[f'price_cluster_density_{window}'] = np.array(padded_density[:len(prices)])
        
        return features

    def _generate_volume_distance_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume-based distance features."""
        features = {}
        
        if 'volume' in data.columns:
            volumes = data['volume'].values
            
            for window in [10, 20, 30]:
                # Rolling volume distances
                volume_distances = []
                for i in range(window, len(volumes)):
                    window_volumes = volumes[i-window:i]
                    if len(window_volumes) > 1:
                        distances = np.abs(np.diff(window_volumes))
                        avg_distance = np.mean(distances)
                        max_distance = np.max(distances)
                        std_distance = np.std(distances)
                        
                        volume_distances.extend([avg_distance, max_distance, std_distance])
                    else:
                        volume_distances.extend([0, 0, 0])
                
                # Pad with zeros for initial values
                padded_distances = [0] * window + volume_distances
                
                features[f'volume_avg_distance_{window}'] = np.array(padded_distances[:len(volumes)])
                features[f'volume_max_distance_{window}'] = np.array(padded_distances[:len(volumes)])
                features[f'volume_std_distance_{window}'] = np.array(padded_distances[:len(volumes)])
                
                # Volume cluster density
                volume_density = []
                for i in range(window, len(volumes)):
                    window_volumes = volumes[i-window:i]
                    if len(window_volumes) > 1:
                        avg_dist = np.mean(np.abs(np.diff(window_volumes)))
                        density = 1 / (avg_dist + 1e-8)
                        volume_density.append(density)
                    else:
                        volume_density.append(0)
                
                padded_density = [0] * window + volume_density
                features[f'volume_cluster_density_{window}'] = np.array(padded_density[:len(volumes)])
        
        return features

    def _generate_volatility_distance_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility-based distance features."""
        features = {}
        
        if 'close' in data.columns:
            returns = data['close'].pct_change().fillna(0).values
            
            for window in [10, 20, 30]:
                # Rolling volatility distances
                vol_distances = []
                for i in range(window, len(returns)):
                    window_returns = returns[i-window:i]
                    if len(window_returns) > 1:
                        # Calculate rolling volatility
                        vol = np.std(window_returns)
                        # Calculate distance from mean volatility
                        mean_vol = np.mean(np.abs(window_returns))
                        distance = abs(vol - mean_vol)
                        vol_distances.append(distance)
                    else:
                        vol_distances.append(0)
                
                # Pad with zeros for initial values
                padded_distances = [0] * window + vol_distances
                features[f'volatility_distance_{window}'] = np.array(padded_distances[:len(returns)])
                
                # Volatility cluster separation
                vol_separation = []
                for i in range(window, len(returns)):
                    window_returns = returns[i-window:i]
                    if len(window_returns) > 1:
                        # Calculate separation as std of volatilities
                        rolling_vols = []
                        for j in range(5, len(window_returns)):
                            sub_window = window_returns[j-5:j]
                            rolling_vols.append(np.std(sub_window))
                        
                        if len(rolling_vols) > 1:
                            separation = np.std(rolling_vols)
                        else:
                            separation = 0
                        vol_separation.append(separation)
                    else:
                        vol_separation.append(0)
                
                padded_separation = [0] * window + vol_separation
                features[f'volatility_separation_{window}'] = np.array(padded_separation[:len(returns)])
        
        return features

    def _generate_cross_feature_distance_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-feature distance features."""
        features = {}
        
        if all(col in data.columns for col in ['close', 'volume']):
            prices = data['close'].values
            volumes = data['volume'].values
            
            for window in [10, 20, 30]:
                # Cross-feature distances
                cross_distances = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    window_volumes = volumes[i-window:i]
                    
                    if len(window_prices) > 1 and len(window_volumes) > 1:
                        # Normalize features
                        norm_prices = (window_prices - np.mean(window_prices)) / (np.std(window_prices) + 1e-8)
                        norm_volumes = (window_volumes - np.mean(window_volumes)) / (np.std(window_volumes) + 1e-8)
                        
                        # Calculate cross-feature distances
                        cross_dist = np.mean(np.abs(norm_prices - norm_volumes))
                        cross_distances.append(cross_dist)
                    else:
                        cross_distances.append(0)
                
                # Pad with zeros for initial values
                padded_distances = [0] * window + cross_distances
                features[f'cross_feature_distance_{window}'] = np.array(padded_distances[:len(prices)])
                
                # Cross-feature correlation distance
                corr_distances = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    window_volumes = volumes[i-window:i]
                    
                    if len(window_prices) > 1 and len(window_volumes) > 1:
                        # Calculate correlation
                        corr = np.corrcoef(window_prices, window_volumes)[0, 1]
                        # Distance from perfect correlation
                        corr_dist = abs(1 - abs(corr))
                        corr_distances.append(corr_dist)
                    else:
                        corr_distances.append(0)
                
                padded_corr_distances = [0] * window + corr_distances
                features[f'cross_correlation_distance_{window}'] = np.array(padded_corr_distances[:len(prices)])
        
        return features


class ClusteringSeparationGenerator(VectorizedFeatureGenerator):
    """
    Clustering Separation Feature Generator.
    
    Generates features that measure cluster separation and boundaries.
    Critical for identifying distinct regime clusters.
    """
    
    def __init__(self):
        base_config = FeatureConfig(
            name="clustering_separation",
            category=FeatureCategory.REGIME,
            description="Cluster separation features for clustering",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=50,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(base_config, enable_matrix_ops=True)
        
        # Initialize VectorBT optimizers
        # vectorbt_optimizer inherited from base class
        self.unified_optimizer = None
        
        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT rolling optimizer
                # vectorbt_rolling_optimizer inherited from base class
                
                # Initialize unified vectorization manager
                unified_config = {
                    "enable_vectorbt": True,
                    "enable_matrix_ops": True,
                    "enable_gpu": False,
                    "optimization_level": "high"
                }
                self.unified_optimizer = UnifiedVectorizationManager(unified_config)
                
                print("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for ClusteringSeparationGenerator")
            except Exception as e:
                print(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate clustering separation features."""
        features = self.generate_features(data)
        
        if features:
            # Return a composite separation score
            separation_score = np.mean(list(features.values()), axis=0)
            return pd.Series(separation_score, index=data.index, name=self.config.name)
        else:
            return pd.Series(0.5, index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive cluster separation features."""
        features = {}
        
        if len(data) < self.config.min_lookback:
            return features
            
        # Regime boundary features
        features.update(self._generate_regime_boundary_features(data))
        
        # Cluster compactness features
        features.update(self._generate_cluster_compactness_features(data))
        
        # Separation strength features
        features.update(self._generate_separation_strength_features(data))
        
        return features

    def _generate_regime_boundary_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime boundary detection features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Regime change points
                change_points = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate regime change strength
                        price_changes = np.diff(window_prices)
                        change_strength = np.std(price_changes)
                        change_points.append(change_strength)
                    else:
                        change_points.append(0)
                
                # Pad with zeros for initial values
                padded_changes = [0] * window + change_points
                features[f'regime_change_strength_{window}'] = np.array(padded_changes[:len(prices)])
                
                # Regime boundary clarity
                boundary_clarity = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate boundary clarity as inverse of transition smoothness
                        price_changes = np.diff(window_prices)
                        smoothness = np.mean(np.abs(price_changes))
                        clarity = 1 / (smoothness + 1e-8)
                        boundary_clarity.append(clarity)
                    else:
                        boundary_clarity.append(0)
                
                padded_clarity = [0] * window + boundary_clarity
                features[f'regime_boundary_clarity_{window}'] = np.array(padded_clarity[:len(prices)])
        
        return features

    def _generate_cluster_compactness_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cluster compactness features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Cluster compactness
                compactness = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate compactness as inverse of variance
                        variance = np.var(window_prices)
                        compactness_score = 1 / (variance + 1e-8)
                        compactness.append(compactness_score)
                    else:
                        compactness.append(0)
                
                # Pad with zeros for initial values
                padded_compactness = [0] * window + compactness
                features[f'cluster_compactness_{window}'] = np.array(padded_compactness[:len(prices)])
                
                # Cluster density
                density = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate density as number of points per unit variance
                        variance = np.var(window_prices)
                        density_score = len(window_prices) / (variance + 1e-8)
                        density.append(density_score)
                    else:
                        density.append(0)
                
                padded_density = [0] * window + density
                features[f'cluster_density_{window}'] = np.array(padded_density[:len(prices)])
        
        return features

    def _generate_separation_strength_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate separation strength features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Separation strength
                separation_strength = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate separation as distance between clusters
                        # Split window into two halves
                        mid = len(window_prices) // 2
                        cluster1 = window_prices[:mid]
                        cluster2 = window_prices[mid:]
                        
                        if len(cluster1) > 0 and len(cluster2) > 0:
                            mean1 = np.mean(cluster1)
                            mean2 = np.mean(cluster2)
                            separation = abs(mean1 - mean2)
                            separation_strength.append(separation)
                        else:
                            separation_strength.append(0)
                    else:
                        separation_strength.append(0)
                
                # Pad with zeros for initial values
                padded_separation = [0] * window + separation_strength
                features[f'separation_strength_{window}'] = np.array(padded_separation[:len(prices)])
                
                # Inter-cluster distance
                inter_cluster_distance = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate inter-cluster distance
                        # Use quantiles to define clusters
                        q33 = np.percentile(window_prices, 33)
                        q67 = np.percentile(window_prices, 67)
                        
                        cluster1 = window_prices[window_prices <= q33]
                        cluster2 = window_prices[window_prices >= q67]
                        
                        if len(cluster1) > 0 and len(cluster2) > 0:
                            mean1 = np.mean(cluster1)
                            mean2 = np.mean(cluster2)
                            inter_dist = abs(mean1 - mean2)
                            inter_cluster_distance.append(inter_dist)
                        else:
                            inter_cluster_distance.append(0)
                    else:
                        inter_cluster_distance.append(0)
                
                padded_inter_dist = [0] * window + inter_cluster_distance
                features[f'inter_cluster_distance_{window}'] = np.array(padded_inter_dist[:len(prices)])
        
        return features


class ClusteringStabilityGenerator(VectorizedFeatureGenerator):
    """
    Clustering Stability Feature Generator.
    
    Generates features that measure clustering stability and consistency.
    Important for robust cluster identification.
    """
    
    def __init__(self):
        base_config = FeatureConfig(
            name="clustering_stability",
            category=FeatureCategory.REGIME,
            description="Clustering stability features",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=50,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(base_config, enable_matrix_ops=True)
        
        # Initialize VectorBT optimizers
        # vectorbt_optimizer inherited from base class
        self.unified_optimizer = None
        
        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT rolling optimizer
                # vectorbt_rolling_optimizer inherited from base class
                
                # Initialize unified vectorization manager
                unified_config = {
                    "enable_vectorbt": True,
                    "enable_matrix_ops": True,
                    "enable_gpu": False,
                    "optimization_level": "high"
                }
                self.unified_optimizer = UnifiedVectorizationManager(unified_config)
                
                print("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for ClusteringStabilityGenerator")
            except Exception as e:
                print(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate clustering stability features."""
        features = self.generate_features(data)
        
        if features:
            # Return a composite stability score
            stability_score = np.mean(list(features.values()), axis=0)
            return pd.Series(stability_score, index=data.index, name=self.config.name)
        else:
            return pd.Series(0.5, index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive clustering stability features."""
        features = {}
        
        if len(data) < self.config.min_lookback:
            return features
            
        # Cluster consistency features
        features.update(self._generate_cluster_consistency_features(data))
        
        # Stability over time features
        features.update(self._generate_temporal_stability_features(data))
        
        # Robustness features
        features.update(self._generate_robustness_features(data))
        
        return features

    def _generate_cluster_consistency_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cluster consistency features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Cluster consistency
                consistency = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate consistency as inverse of coefficient of variation
                        mean_price = np.mean(window_prices)
                        std_price = np.std(window_prices)
                        cv = std_price / (mean_price + 1e-8)
                        consistency_score = 1 / (cv + 1e-8)
                        consistency.append(consistency_score)
                    else:
                        consistency.append(0)
                
                # Pad with zeros for initial values
                padded_consistency = [0] * window + consistency
                features[f'cluster_consistency_{window}'] = np.array(padded_consistency[:len(prices)])
                
                # Cluster homogeneity
                homogeneity = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate homogeneity as inverse of range
                        price_range = np.max(window_prices) - np.min(window_prices)
                        mean_price = np.mean(window_prices)
                        normalized_range = price_range / (mean_price + 1e-8)
                        homogeneity_score = 1 / (normalized_range + 1e-8)
                        homogeneity.append(homogeneity_score)
                    else:
                        homogeneity.append(0)
                
                padded_homogeneity = [0] * window + homogeneity
                features[f'cluster_homogeneity_{window}'] = np.array(padded_homogeneity[:len(prices)])
        
        return features

    def _generate_temporal_stability_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate temporal stability features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Temporal stability
                temporal_stability = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate temporal stability as autocorrelation
                        price_changes = np.diff(window_prices)
                        if len(price_changes) > 1:
                            autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
                            if not np.isnan(autocorr):
                                temporal_stability.append(abs(autocorr))
                            else:
                                temporal_stability.append(0)
                        else:
                            temporal_stability.append(0)
                    else:
                        temporal_stability.append(0)
                
                # Pad with zeros for initial values
                padded_stability = [0] * window + temporal_stability
                features[f'temporal_stability_{window}'] = np.array(padded_stability[:len(prices)])
                
                # Regime persistence
                regime_persistence = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate regime persistence
                        price_changes = np.diff(window_prices)
                        positive_changes = np.sum(price_changes > 0)
                        negative_changes = np.sum(price_changes < 0)
                        total_changes = len(price_changes)
                        
                        if total_changes > 0:
                            persistence = max(positive_changes, negative_changes) / total_changes
                            regime_persistence.append(persistence)
                        else:
                            regime_persistence.append(0)
                    else:
                        regime_persistence.append(0)
                
                padded_persistence = [0] * window + regime_persistence
                features[f'regime_persistence_{window}'] = np.array(padded_persistence[:len(prices)])
        
        return features

    def _generate_robustness_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate robustness features."""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            for window in [10, 20, 30]:
                # Robustness to outliers
                robustness = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate robustness as inverse of outlier impact
                        mean_price = np.mean(window_prices)
                        std_price = np.std(window_prices)
                        
                        # Count outliers (beyond 2 standard deviations)
                        outliers = np.sum(np.abs(window_prices - mean_price) > 2 * std_price)
                        outlier_ratio = outliers / len(window_prices)
                        robustness_score = 1 - outlier_ratio
                        robustness.append(robustness_score)
                    else:
                        robustness.append(0)
                
                # Pad with zeros for initial values
                padded_robustness = [0] * window + robustness
                features[f'cluster_robustness_{window}'] = np.array(padded_robustness[:len(prices)])
                
                # Noise resistance
                noise_resistance = []
                for i in range(window, len(prices)):
                    window_prices = prices[i-window:i]
                    if len(window_prices) > 1:
                        # Calculate noise resistance as signal-to-noise ratio
                        signal = np.var(window_prices)
                        # Estimate noise as high-frequency component
                        price_changes = np.diff(window_prices)
                        noise = np.var(price_changes)
                        
                        if noise > 0:
                            snr = signal / noise
                            noise_resistance.append(snr)
                        else:
                            noise_resistance.append(0)
                    else:
                        noise_resistance.append(0)
                
                padded_noise_resistance = [0] * window + noise_resistance
                features[f'noise_resistance_{window}'] = np.array(padded_noise_resistance[:len(prices)])
        
        return features


class ClusteringIntegration(VectorizedFeatureGenerator):
    """
    Unified Clustering Feature Generator.
    
    Integrates all clustering-specific feature generators for optimal
    clustering algorithm performance. These features should NEVER be used
    during live trading.
    """
    
    def __init__(self, config: Optional[ClusteringFeatureConfig] = None):
        if config is None:
            config = ClusteringFeatureConfig()
        
        self.clustering_config = config
        self.config = config
        
        # Initialize base config
        base_config = FeatureConfig(
            name="clustering_integration",
            category=FeatureCategory.REGIME,
            description="Unified clustering features for regime discovery",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=50,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        
        super().__init__(base_config, enable_matrix_ops=True)
        
        # Initialize VectorBT optimizers
        # vectorbt_optimizer inherited from base class
        self.unified_optimizer = None
        
        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT rolling optimizer
                # vectorbt_rolling_optimizer inherited from base class
                
                # Initialize unified vectorization manager
                unified_config = {
                    "enable_vectorbt": True,
                    "enable_matrix_ops": True,
                    "enable_gpu": False,
                    "optimization_level": "high"
                }
                self.unified_optimizer = UnifiedVectorizationManager(unified_config)
                
                print("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for ClusteringIntegration")
            except Exception as e:
                print(f"⚠️ VectorBT optimizer initialization failed: {e}")
        
        # Initialize clustering feature generators
        self.distance_generator = ClusteringDistanceGenerator() if config.include_distance_features else None
        self.separation_generator = ClusteringSeparationGenerator() if config.include_separation_features else None
        self.stability_generator = ClusteringStabilityGenerator() if config.include_stability_features else None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate unified clustering features."""
        features = self.generate_features(data)
        
        if features:
            # Return a composite clustering score
            clustering_score = np.mean(list(features.values()), axis=0)
            return pd.Series(clustering_score, index=data.index, name=self.config.name)
        else:
            return pd.Series(0.5, index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive clustering features."""
        tprint_info("🎯 Generating comprehensive clustering features")
        tprint_data_preview(data, "Input Data for Clustering Features", max_rows=2, max_cols=5)
        
        features = {}
        
        if len(data) < self.config.min_lookback:
            tprint_warning(f"⚠️ Insufficient data: {len(data)} < {self.config.min_lookback}")
            return features
        
        with tprint_timer("Comprehensive Clustering Feature Generation", level="PERFORMANCE"):
            # Generate features from each generator
            if self.distance_generator:
                tprint_info("📊 Generating distance features")
                distance_features = self.distance_generator.generate_features(data)
                features.update(distance_features)
                tprint_info(f"✅ Generated {len(distance_features)} distance features")
            
            if self.separation_generator:
                tprint_info("📊 Generating separation features")
                separation_features = self.separation_generator.generate_features(data)
                features.update(separation_features)
                tprint_info(f"✅ Generated {len(separation_features)} separation features")
            
            if self.stability_generator:
                tprint_info("📊 Generating stability features")
                stability_features = self.stability_generator.generate_features(data)
                features.update(stability_features)
                tprint_info(f"✅ Generated {len(stability_features)} stability features")
        
        tprint_success(f"🎉 Generated {len(features)} total comprehensive clustering features")
        return features


# Convenience functions
def create_clustering_feature_generators() -> List[FeatureGenerator]:
    """Create all clustering feature generators."""
    tprint_info("🏭 Creating clustering feature generators")
    
    generators = [
        ClusteringDistanceGenerator(),
        ClusteringSeparationGenerator(),
        ClusteringStabilityGenerator(),
        ClusteringIntegration()
    ]
    
    tprint_success(f"✅ Created {len(generators)} clustering feature generators")
    return generators


def create_default_clustering_generators() -> List[FeatureGenerator]:
    """Create default clustering feature generators."""
    return [
        ClusteringDistanceGenerator(),
        ClusteringSeparationGenerator(),
        ClusteringStabilityGenerator()
    ]


def generate_clustering_features(data: pd.DataFrame,
                               config: Optional[ClusteringFeatureConfig] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate clustering-focused features.
    
    Args:
        data: Market data DataFrame with OHLCV columns
        config: Configuration for clustering feature generation
    
    Returns:
        Tuple of (features_dict, summary_dict)
    """
    if config is None:
        config = ClusteringFeatureConfig()
    
    generator = ClusteringIntegration(config)
    features = generator.generate_features(data)
    
    # Create summary
    summary = {
        'total_features': len(features),
        'feature_names': list(features.keys()),
        'config': config
    }
    
    return features, summary


__all__ = [
    'ClusteringDistanceGenerator',
    'ClusteringSeparationGenerator',
    'ClusteringStabilityGenerator',
    'ClusteringIntegration',
    'ClusteringFeatureConfig',
    'create_clustering_feature_generators',
    'create_default_clustering_generators',
    'generate_clustering_features'
]