"""
Cluster Distinctiveness Metrics for Feature Selection

This module provides comprehensive metrics to evaluate how well features
separate clusters in regime clustering tasks. It implements various
statistical measures to assess feature quality for clustering.

Key Metrics:
- F-ratio: Between-cluster variance vs within-cluster variance
- Silhouette-based feature importance
- Cluster separation strength
- Inter-cluster distance measures
- Cluster compactness metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some metrics will be disabled.")

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.records import Drawdowns
    from vectorbt.portfolio import Portfolio
    from vectorbt.indicators import RSI, MACD, BollingerBands
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available. Some optimizations will be disabled.")

# Hardware acceleration imports
try:
    from src.utils.hardware import get_hardware_accelerator, HardwareAccelerator
    from src.utils.hardware.gpu_acceleration import GPUAccelerator
    from src.utils.hardware.vectorization import VectorizationManager
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    warnings.warn("Hardware acceleration not available. Using CPU-only computations.")

# VectorBT rolling optimizer imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_optimization_system import UnifiedVectorizationManager
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    warnings.warn("VectorBT optimization not available. Using standard computations.")


@dataclass
class ClusterDistinctivenessConfig:
    """Configuration for cluster distinctiveness metrics."""
    
    # Minimum cluster size for valid calculations
    min_cluster_size: int = 3
    
    # Minimum number of clusters for valid calculations
    min_clusters: int = 2
    
    # Noise label (typically -1 for DBSCAN)
    noise_label: int = -1
    
    # Enable advanced metrics (requires scikit-learn)
    enable_advanced_metrics: bool = True
    
    # Scaling for feature values before calculation
    enable_scaling: bool = True
    
    # Minimum variance threshold for feature validity
    min_variance_threshold: float = 1e-8
    
    # Performance optimization settings
    enable_fast_proxies: bool = True  # Use fast proxy calculations
    max_samples_for_advanced: int = 10000  # Limit samples for expensive calculations
    enable_caching: bool = True  # Cache intermediate calculations
    batch_size: int = 1000  # Process features in batches
    
    # Approximation settings
    use_approximate_silhouette: bool = True  # Use faster silhouette approximation
    silhouette_sample_ratio: float = 0.1  # Sample ratio for silhouette calculation
    
    # VectorBT optimization settings
    enable_vectorbt_optimization: bool = True  # Enable VectorBT optimizations
    enable_hardware_acceleration: bool = True  # Enable hardware acceleration
    use_gpu: bool = False  # Use GPU acceleration if available
    vectorbt_chunk_size: int = 10000  # Chunk size for VectorBT operations
    
    # Unified optimization settings
    enable_unified_optimization: bool = True  # Enable unified optimization system
    memory_limit_gb: float = 8.0  # Memory limit for operations
    enable_parallel_processing: bool = True  # Enable parallel processing


class ClusterDistinctivenessCalculator:
    """
    Calculator for cluster distinctiveness metrics.
    
    This class provides various metrics to evaluate how well individual
    features or feature combinations separate clusters in regime clustering.
    """
    
    def __init__(self, config: Optional[ClusterDistinctivenessConfig] = None):
        self.config = config or ClusterDistinctivenessConfig()
        self.scaler = StandardScaler() if self.config.enable_scaling else None
        
        # Performance optimization caches
        self._cluster_stats_cache = {} if self.config.enable_caching else None
        self._feature_stats_cache = {} if self.config.enable_caching else None
        
        # Initialize VectorBT optimizers
        self.vectorbt_optimizer = None
        self.unified_optimizer = None
        self.hardware_accelerator = None
        
        if self.config.enable_vectorbt_optimization and VECTORBT_AVAILABLE:
            self._initialize_vectorbt_optimizers()
        
        if self.config.enable_hardware_acceleration and HARDWARE_AVAILABLE:
            self._initialize_hardware_accelerators()
    
    def _initialize_vectorbt_optimizers(self):
        """Initialize VectorBT optimizers."""
        try:
            if OPTIMIZATION_AVAILABLE:
                # Initialize VectorBT Rolling Optimizer
                self.vectorbt_optimizer = VectorBTRollingOptimizer(
                    enable_gpu=self.config.use_gpu,
                    enable_parallel=self.config.enable_parallel_processing,
                    memory_efficient=True,
                    chunk_size=self.config.vectorbt_chunk_size
                )
                
                # Initialize Unified Optimization System
                from src.feature_generation.utils.unified_optimization_system import UnifiedOptimizationConfig
                unified_config = UnifiedOptimizationConfig(
                    enable_normalization=True,
                    enable_scaling=True,
                    enable_vectorization=True,
                    enable_hardware_optimization=self.config.use_gpu,
                    memory_limit_gb=self.config.memory_limit_gb
                )
                self.unified_optimizer = UnifiedVectorizationManager(unified_config)
                
                print("✅ VectorBT optimizers initialized successfully")
            else:
                print("⚠️ VectorBT optimization not available")
        except Exception as e:
            print(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    def _initialize_hardware_accelerators(self):
        """Initialize hardware accelerators."""
        try:
            self.hardware_accelerator = get_hardware_accelerator(
                use_gpu=self.config.use_gpu,
                enable_parallel=self.config.enable_parallel_processing
            )
            print(f"✅ Hardware accelerator initialized: {type(self.hardware_accelerator).__name__}")
        except Exception as e:
            print(f"⚠️ Hardware accelerator initialization failed: {e}")
            self.hardware_accelerator = None
    
    def calculate_feature_distinctiveness(self, 
                                        features: Dict[str, np.ndarray], 
                                        cluster_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate distinctiveness metrics for each feature - OPTIMIZED with batch processing.
        
        Args:
            features: Dictionary of feature names to feature values
            cluster_labels: Cluster labels for each sample
            
        Returns:
            Dictionary mapping feature names to their distinctiveness metrics
        """
        if not features or len(cluster_labels) == 0:
            return {}
        
        # Validate cluster labels
        unique_clusters = [c for c in set(cluster_labels) if c != self.config.noise_label]
        if len(unique_clusters) < self.config.min_clusters:
            warnings.warn(f"Insufficient clusters for distinctiveness calculation: {len(unique_clusters)}")
            return {}
        
        results = {}
        
        # Batch processing for large feature sets
        feature_items = list(features.items())
        batch_size = self.config.batch_size
        
        for i in range(0, len(feature_items), batch_size):
            batch = feature_items[i:i + batch_size]
            
            # Process batch
            for feature_name, feature_values in batch:
                try:
                    # Validate feature
                    if not self._validate_feature(feature_values):
                        results[feature_name] = self._get_zero_metrics()
                        continue
                    
                    # Calculate distinctiveness metrics
                    metrics = self._calculate_single_feature_distinctiveness(
                        feature_values, cluster_labels
                    )
                    results[feature_name] = metrics
                    
                except Exception as e:
                    warnings.warn(f"Failed to calculate distinctiveness for {feature_name}: {e}")
                    results[feature_name] = self._get_zero_metrics()
        
        return results
    
    def _calculate_single_feature_distinctiveness(self, 
                                                feature_values: np.ndarray, 
                                                cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate distinctiveness metrics for a single feature."""
        metrics = {}
        
        # Basic F-ratio
        metrics['f_ratio'] = self._calculate_f_ratio(feature_values, cluster_labels)
        
        # Cluster separation strength
        metrics['separation_strength'] = self._calculate_separation_strength(
            feature_values, cluster_labels
        )
        
        # Inter-cluster distance
        metrics['inter_cluster_distance'] = self._calculate_inter_cluster_distance(
            feature_values, cluster_labels
        )
        
        # Cluster compactness
        metrics['cluster_compactness'] = self._calculate_cluster_compactness(
            feature_values, cluster_labels
        )
        
        # Advanced metrics (if sklearn available)
        if SKLEARN_AVAILABLE and self.config.enable_advanced_metrics:
            metrics.update(self._calculate_advanced_metrics(feature_values, cluster_labels))
        
        # Combined distinctiveness score
        metrics['combined_score'] = self._calculate_combined_score(metrics)
        
        return metrics
    
    def _calculate_f_ratio(self, feature_values: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate F-ratio (between-cluster variance / within-cluster variance) - OPTIMIZED."""
        # Use cached cluster stats if available
        cache_key = f"cluster_stats_{hash(cluster_labels.tobytes())}"
        if self._cluster_stats_cache and cache_key in self._cluster_stats_cache:
            cluster_stats = self._cluster_stats_cache[cache_key]
        else:
            cluster_stats = self._get_cluster_stats_optimized(cluster_labels)
            if self._cluster_stats_cache is not None:
                self._cluster_stats_cache[cache_key] = cluster_stats
        
        unique_clusters = cluster_stats['unique_clusters']
        if len(unique_clusters) < 2:
            return 0.0
        
        # Fast F-ratio calculation using vectorized operations
        if self.config.enable_fast_proxies:
            return self._calculate_f_ratio_fast(feature_values, cluster_labels, cluster_stats)
        
        # Original calculation for accuracy
        cluster_means = []
        cluster_sizes = []
        cluster_values = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_vals = feature_values[cluster_mask]
            
            if len(cluster_vals) >= self.config.min_cluster_size:
                cluster_means.append(np.mean(cluster_vals))
                cluster_sizes.append(len(cluster_vals))
                cluster_values.append(cluster_vals)
        
        if len(cluster_means) < 2:
            return 0.0
        
        # Overall mean
        overall_mean = np.mean(feature_values)
        
        # Between-cluster variance
        between_var = sum(size * (mean - overall_mean)**2 
                         for mean, size in zip(cluster_means, cluster_sizes))
        
        # Within-cluster variance
        within_var = 0
        for i, cluster_id in enumerate(unique_clusters):
            if i < len(cluster_values):
                cluster_mean = cluster_means[i]
                within_var += np.sum((cluster_values[i] - cluster_mean)**2)
        
        # F-ratio
        if within_var > 0:
            f_ratio = between_var / within_var
        else:
            f_ratio = 0.0
        
        return float(f_ratio)
    
    def _get_cluster_stats_optimized(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Get cluster statistics with caching and optimization."""
        unique_clusters = [c for c in set(cluster_labels) if c != self.config.noise_label]
        cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
        
        return {
            'unique_clusters': unique_clusters,
            'cluster_sizes': cluster_sizes,
            'total_samples': len(cluster_labels),
            'n_clusters': len(unique_clusters)
        }
    
    def _calculate_f_ratio_fast(self, feature_values: np.ndarray, cluster_labels: np.ndarray, 
                               cluster_stats: Dict[str, Any]) -> float:
        """Fast F-ratio calculation using VectorBT and hardware acceleration."""
        unique_clusters = cluster_stats['unique_clusters']
        
        # Use VectorBT optimization if available
        if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
            return self._calculate_f_ratio_vectorbt(feature_values, cluster_labels, unique_clusters)
        
        # Use hardware acceleration if available
        if self.hardware_accelerator:
            return self._calculate_f_ratio_hardware_accelerated(feature_values, cluster_labels, unique_clusters)
        
        # Fallback to standard vectorized calculation
        return self._calculate_f_ratio_standard_vectorized(feature_values, cluster_labels, unique_clusters)
    
    def _calculate_f_ratio_vectorbt(self, feature_values: np.ndarray, cluster_labels: np.ndarray, 
                                   unique_clusters: List[int]) -> float:
        """F-ratio calculation using VectorBT optimizations."""
        try:
            # Convert to pandas Series for VectorBT
            feature_series = pd.Series(feature_values)
            cluster_series = pd.Series(cluster_labels)
            
            # Use VectorBT rolling operations for cluster statistics
            cluster_means = []
            cluster_sizes = []
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_series == cluster_id
                cluster_vals = feature_series[cluster_mask]
                
                if len(cluster_vals) >= self.config.min_cluster_size:
                    # Use VectorBT for efficient mean calculation
                    cluster_mean = vbt.rolling_mean(cluster_vals, window=len(cluster_vals)).iloc[-1]
                    cluster_means.append(cluster_mean)
                    cluster_sizes.append(len(cluster_vals))
            
            if len(cluster_means) < 2:
                return 0.0
            
            cluster_means = np.array(cluster_means)
            cluster_sizes = np.array(cluster_sizes)
            
            # Overall mean using VectorBT
            overall_mean = vbt.rolling_mean(feature_series, window=len(feature_series)).iloc[-1]
            
            # Vectorized between-cluster variance
            between_var = np.sum(cluster_sizes * (cluster_means - overall_mean)**2)
            
            # Calculate within-cluster variance using VectorBT
            within_var = 0
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_series == cluster_id
                cluster_vals = feature_series[cluster_mask]
                
                if len(cluster_vals) >= self.config.min_cluster_size:
                    # Use VectorBT for variance calculation
                    cluster_var = vbt.rolling_std(cluster_vals, window=len(cluster_vals)).iloc[-1]**2
                    within_var += cluster_var * len(cluster_vals)
            
            # F-ratio
            if within_var > 0:
                f_ratio = between_var / within_var
            else:
                f_ratio = 0.0
            
            return float(f_ratio)
            
        except Exception as e:
            print(f"⚠️ VectorBT F-ratio calculation failed: {e}")
            return self._calculate_f_ratio_standard_vectorized(feature_values, cluster_labels, unique_clusters)
    
    def _calculate_f_ratio_hardware_accelerated(self, feature_values: np.ndarray, cluster_labels: np.ndarray, 
                                               unique_clusters: List[int]) -> float:
        """F-ratio calculation using hardware acceleration."""
        try:
            # Use hardware accelerator for cluster statistics
            cluster_stats = self.hardware_accelerator.calculate_cluster_statistics(
                feature_values, cluster_labels, unique_clusters
            )
            
            cluster_means = cluster_stats['means']
            cluster_sizes = cluster_stats['sizes']
            
            # Filter by minimum cluster size
            valid_mask = cluster_sizes >= self.config.min_cluster_size
            if np.sum(valid_mask) < 2:
                return 0.0
            
            cluster_means = cluster_means[valid_mask]
            cluster_sizes = cluster_sizes[valid_mask]
            
            # Overall mean
            overall_mean = np.mean(feature_values)
            
            # Vectorized between-cluster variance
            between_var = np.sum(cluster_sizes * (cluster_means - overall_mean)**2)
            
            # Calculate within-cluster variance using hardware acceleration
            within_var = self.hardware_accelerator.calculate_within_cluster_variance(
                feature_values, cluster_labels, cluster_means, unique_clusters
            )
            
            # F-ratio
            if within_var > 0:
                f_ratio = between_var / within_var
            else:
                f_ratio = 0.0
            
            return float(f_ratio)
            
        except Exception as e:
            print(f"⚠️ Hardware accelerated F-ratio calculation failed: {e}")
            return self._calculate_f_ratio_standard_vectorized(feature_values, cluster_labels, unique_clusters)
    
    def _calculate_f_ratio_standard_vectorized(self, feature_values: np.ndarray, cluster_labels: np.ndarray, 
                                             unique_clusters: List[int]) -> float:
        """Standard vectorized F-ratio calculation."""
        # Vectorized cluster mean calculation
        cluster_means = np.array([np.mean(feature_values[cluster_labels == c]) 
                                 for c in unique_clusters])
        cluster_sizes = np.array([np.sum(cluster_labels == c) 
                                 for c in unique_clusters])
        
        # Filter by minimum cluster size
        valid_mask = cluster_sizes >= self.config.min_cluster_size
        if np.sum(valid_mask) < 2:
            return 0.0
        
        cluster_means = cluster_means[valid_mask]
        cluster_sizes = cluster_sizes[valid_mask]
        
        # Overall mean
        overall_mean = np.mean(feature_values)
        
        # Vectorized between-cluster variance
        between_var = np.sum(cluster_sizes * (cluster_means - overall_mean)**2)
        
        # Vectorized within-cluster variance
        within_var = 0
        for i, cluster_id in enumerate(unique_clusters):
            if valid_mask[i]:
                cluster_mask = cluster_labels == cluster_id
                cluster_vals = feature_values[cluster_mask]
                cluster_mean = cluster_means[i]
                within_var += np.sum((cluster_vals - cluster_mean)**2)
        
        # F-ratio
        if within_var > 0:
            f_ratio = between_var / within_var
        else:
            f_ratio = 0.0
        
        return float(f_ratio)
    
    def _calculate_separation_strength(self, feature_values: np.ndarray, 
                                     cluster_labels: np.ndarray) -> float:
        """Calculate cluster separation strength using VectorBT optimizations."""
        unique_clusters = [c for c in set(cluster_labels) if c != self.config.noise_label]
        
        if len(unique_clusters) < 2:
            return 0.0
        
        # Use VectorBT optimization if available
        if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
            return self._calculate_separation_strength_vectorbt(feature_values, cluster_labels, unique_clusters)
        
        # Use hardware acceleration if available
        if self.hardware_accelerator:
            return self._calculate_separation_strength_hardware_accelerated(feature_values, cluster_labels, unique_clusters)
        
        # Fallback to standard calculation
        return self._calculate_separation_strength_standard(feature_values, cluster_labels, unique_clusters)
    
    def _calculate_separation_strength_vectorbt(self, feature_values: np.ndarray, 
                                              cluster_labels: np.ndarray, 
                                              unique_clusters: List[int]) -> float:
        """Separation strength calculation using VectorBT."""
        try:
            # Convert to pandas Series for VectorBT
            feature_series = pd.Series(feature_values)
            cluster_series = pd.Series(cluster_labels)
            
            # Calculate cluster means using VectorBT
            cluster_means = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_series == cluster_id
                cluster_vals = feature_series[cluster_mask]
                
                if len(cluster_vals) >= self.config.min_cluster_size:
                    cluster_mean = vbt.rolling_mean(cluster_vals, window=len(cluster_vals)).iloc[-1]
                    cluster_means.append(cluster_mean)
            
            if len(cluster_means) < 2:
                return 0.0
            
            # Calculate inter-cluster distance using VectorBT
            cluster_means_series = pd.Series(cluster_means)
            inter_cluster_dist = vbt.rolling_std(cluster_means_series, window=len(cluster_means_series)).iloc[-1]
            
            # Calculate intra-cluster spreads using VectorBT
            intra_cluster_spreads = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_series == cluster_id
                cluster_vals = feature_series[cluster_mask]
                
                if len(cluster_vals) >= self.config.min_cluster_size:
                    cluster_std = vbt.rolling_std(cluster_vals, window=len(cluster_vals)).iloc[-1]
                    intra_cluster_spreads.append(cluster_std)
            
            if not intra_cluster_spreads:
                return 0.0
            
            # Calculate average intra-cluster spread using VectorBT
            spreads_series = pd.Series(intra_cluster_spreads)
            avg_intra_cluster_spread = vbt.rolling_mean(spreads_series, window=len(spreads_series)).iloc[-1]
            
            # Separation strength
            if avg_intra_cluster_spread > 0:
                separation_strength = inter_cluster_dist / avg_intra_cluster_spread
            else:
                separation_strength = 0.0
            
            return float(separation_strength)
            
        except Exception as e:
            print(f"⚠️ VectorBT separation strength calculation failed: {e}")
            return self._calculate_separation_strength_standard(feature_values, cluster_labels, unique_clusters)
    
    def _calculate_separation_strength_hardware_accelerated(self, feature_values: np.ndarray, 
                                                          cluster_labels: np.ndarray, 
                                                          unique_clusters: List[int]) -> float:
        """Separation strength calculation using hardware acceleration."""
        try:
            # Use hardware accelerator for cluster statistics
            cluster_stats = self.hardware_accelerator.calculate_cluster_statistics(
                feature_values, cluster_labels, unique_clusters
            )
            
            cluster_means = cluster_stats['means']
            cluster_stds = cluster_stats['stds']
            
            # Filter by minimum cluster size
            valid_mask = cluster_stats['sizes'] >= self.config.min_cluster_size
            if np.sum(valid_mask) < 2:
                return 0.0
            
            cluster_means = cluster_means[valid_mask]
            cluster_stds = cluster_stds[valid_mask]
            
            # Calculate inter-cluster distance
            inter_cluster_dist = np.std(cluster_means)
            
            # Calculate average intra-cluster spread
            avg_intra_cluster_spread = np.mean(cluster_stds)
            
            # Separation strength
            if avg_intra_cluster_spread > 0:
                separation_strength = inter_cluster_dist / avg_intra_cluster_spread
            else:
                separation_strength = 0.0
            
            return float(separation_strength)
            
        except Exception as e:
            print(f"⚠️ Hardware accelerated separation strength calculation failed: {e}")
            return self._calculate_separation_strength_standard(feature_values, cluster_labels, unique_clusters)
    
    def _calculate_separation_strength_standard(self, feature_values: np.ndarray, 
                                              cluster_labels: np.ndarray, 
                                              unique_clusters: List[int]) -> float:
        """Standard separation strength calculation."""
        # Calculate cluster means
        cluster_means = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_vals = feature_values[cluster_mask]
            
            if len(cluster_vals) >= self.config.min_cluster_size:
                cluster_means.append(np.mean(cluster_vals))
        
        if len(cluster_means) < 2:
            return 0.0
        
        # Calculate separation as ratio of inter-cluster distance to intra-cluster spread
        inter_cluster_dist = np.std(cluster_means)
        
        # Calculate average intra-cluster spread
        intra_cluster_spreads = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_vals = feature_values[cluster_mask]
            
            if len(cluster_vals) >= self.config.min_cluster_size:
                intra_cluster_spreads.append(np.std(cluster_vals))
        
        if not intra_cluster_spreads:
            return 0.0
        
        avg_intra_cluster_spread = np.mean(intra_cluster_spreads)
        
        # Separation strength
        if avg_intra_cluster_spread > 0:
            separation_strength = inter_cluster_dist / avg_intra_cluster_spread
        else:
            separation_strength = 0.0
        
        return float(separation_strength)
    
    def _calculate_inter_cluster_distance(self, feature_values: np.ndarray, 
                                        cluster_labels: np.ndarray) -> float:
        """Calculate average inter-cluster distance."""
        unique_clusters = [c for c in set(cluster_labels) if c != self.config.noise_label]
        
        if len(unique_clusters) < 2:
            return 0.0
        
        # Calculate cluster means
        cluster_means = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_vals = feature_values[cluster_mask]
            
            if len(cluster_vals) >= self.config.min_cluster_size:
                cluster_means.append(np.mean(cluster_vals))
        
        if len(cluster_means) < 2:
            return 0.0
        
        # Calculate pairwise distances between cluster means
        distances = []
        for i in range(len(cluster_means)):
            for j in range(i + 1, len(cluster_means)):
                distances.append(abs(cluster_means[i] - cluster_means[j]))
        
        return float(np.mean(distances)) if distances else 0.0
    
    def _calculate_cluster_compactness(self, feature_values: np.ndarray, 
                                     cluster_labels: np.ndarray) -> float:
        """Calculate cluster compactness (inverse of average intra-cluster spread)."""
        unique_clusters = [c for c in set(cluster_labels) if c != self.config.noise_label]
        
        if not unique_clusters:
            return 0.0
        
        # Calculate intra-cluster spreads
        intra_cluster_spreads = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_vals = feature_values[cluster_mask]
            
            if len(cluster_vals) >= self.config.min_cluster_size:
                intra_cluster_spreads.append(np.std(cluster_vals))
        
        if not intra_cluster_spreads:
            return 0.0
        
        # Compactness is inverse of average spread
        avg_spread = np.mean(intra_cluster_spreads)
        compactness = 1.0 / (1.0 + avg_spread) if avg_spread > 0 else 0.0
        
        return float(compactness)
    
    def _calculate_advanced_metrics(self, feature_values: np.ndarray, 
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate advanced metrics using scikit-learn - OPTIMIZED."""
        if not SKLEARN_AVAILABLE:
            return {}
        
        unique_clusters = [c for c in set(cluster_labels) if c != self.config.noise_label]
        
        if len(unique_clusters) < 2:
            return {}
        
        # Filter out noise points
        valid_mask = cluster_labels != self.config.noise_label
        valid_values = feature_values[valid_mask]
        valid_labels = cluster_labels[valid_mask]
        
        if len(valid_values) < 4:  # Need at least 4 points for silhouette
            return {}
        
        # Performance optimization: limit samples for expensive calculations
        if len(valid_values) > self.config.max_samples_for_advanced:
            # Sample data for expensive calculations
            sample_size = min(self.config.max_samples_for_advanced, len(valid_values))
            sample_indices = np.random.choice(len(valid_values), sample_size, replace=False)
            valid_values = valid_values[sample_indices]
            valid_labels = valid_labels[sample_indices]
        
        # Scale values if enabled
        if self.scaler is not None:
            try:
                valid_values_scaled = self.scaler.fit_transform(valid_values.reshape(-1, 1)).flatten()
            except:
                valid_values_scaled = valid_values
        else:
            valid_values_scaled = valid_values
        
        metrics = {}
        
        # Use fast proxies for expensive calculations
        if self.config.enable_fast_proxies:
            metrics.update(self._calculate_advanced_metrics_fast(valid_values_scaled, valid_labels))
        else:
            metrics.update(self._calculate_advanced_metrics_full(valid_values_scaled, valid_labels))
        
        return metrics
    
    def _calculate_advanced_metrics_fast(self, valid_values: np.ndarray, 
                                       valid_labels: np.ndarray) -> Dict[str, float]:
        """Fast approximation of advanced metrics."""
        metrics = {}
        
        try:
            # Fast silhouette approximation using sampling
            if self.config.use_approximate_silhouette and len(valid_values) > 100:
                sample_size = max(50, int(len(valid_values) * self.config.silhouette_sample_ratio))
                sample_indices = np.random.choice(len(valid_values), sample_size, replace=False)
                sample_values = valid_values[sample_indices]
                sample_labels = valid_labels[sample_indices]
                
                if len(set(sample_labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(
                        sample_values.reshape(-1, 1), sample_labels
                    )
                else:
                    metrics['silhouette_score'] = 0.0
            else:
                # Use full calculation for small datasets
                if len(set(valid_labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(
                        valid_values.reshape(-1, 1), valid_labels
                    )
                else:
                    metrics['silhouette_score'] = 0.0
        except:
            metrics['silhouette_score'] = 0.0
        
        try:
            # Calinski-Harabasz score (relatively fast)
            if len(set(valid_labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    valid_values.reshape(-1, 1), valid_labels
                )
            else:
                metrics['calinski_harabasz_score'] = 0.0
        except:
            metrics['calinski_harabasz_score'] = 0.0
        
        try:
            # Davies-Bouldin score (relatively fast)
            if len(set(valid_labels)) > 1:
                db_score = davies_bouldin_score(
                    valid_values.reshape(-1, 1), valid_labels
                )
                metrics['davies_bouldin_score'] = 1.0 / (1.0 + db_score)  # Invert for higher = better
            else:
                metrics['davies_bouldin_score'] = 0.0
        except:
            metrics['davies_bouldin_score'] = 0.0
        
        return metrics
    
    def _calculate_advanced_metrics_full(self, valid_values: np.ndarray, 
                                       valid_labels: np.ndarray) -> Dict[str, float]:
        """Full calculation of advanced metrics."""
        metrics = {}
        
        try:
            # Silhouette score
            if len(set(valid_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(
                    valid_values.reshape(-1, 1), valid_labels
                )
            else:
                metrics['silhouette_score'] = 0.0
        except:
            metrics['silhouette_score'] = 0.0
        
        try:
            # Calinski-Harabasz score
            if len(set(valid_labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    valid_values.reshape(-1, 1), valid_labels
                )
            else:
                metrics['calinski_harabasz_score'] = 0.0
        except:
            metrics['calinski_harabasz_score'] = 0.0
        
        try:
            # Davies-Bouldin score (lower is better, so we invert it)
            if len(set(valid_labels)) > 1:
                db_score = davies_bouldin_score(
                    valid_values.reshape(-1, 1), valid_labels
                )
                metrics['davies_bouldin_score'] = 1.0 / (1.0 + db_score)  # Invert for higher = better
            else:
                metrics['davies_bouldin_score'] = 0.0
        except:
            metrics['davies_bouldin_score'] = 0.0
        
        return metrics
    
    def _calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """Calculate combined distinctiveness score from individual metrics."""
        # Weights for different metrics
        weights = {
            'f_ratio': 0.3,
            'separation_strength': 0.25,
            'inter_cluster_distance': 0.15,
            'cluster_compactness': 0.15,
            'silhouette_score': 0.1,
            'calinski_harabasz_score': 0.05
        }
        
        combined_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                # Normalize scores to 0-1 range
                normalized_score = min(1.0, max(0.0, metrics[metric_name]))
                combined_score += weight * normalized_score
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_score /= total_weight
        
        return float(combined_score)
    
    def _validate_feature(self, feature_values: np.ndarray) -> bool:
        """Validate feature for distinctiveness calculation."""
        # Check for sufficient data
        if len(feature_values) == 0:
            return False
        
        # Check for sufficient variance
        if np.var(feature_values) < self.config.min_variance_threshold:
            return False
        
        # Check for excessive NaN values
        nan_ratio = np.isnan(feature_values).sum() / len(feature_values)
        if nan_ratio > 0.5:
            return False
        
        # Check for constant values
        if len(np.unique(feature_values)) < 3:
            return False
        
        return True
    
    def _get_zero_metrics(self) -> Dict[str, float]:
        """Return zero metrics for invalid features."""
        return {
            'f_ratio': 0.0,
            'separation_strength': 0.0,
            'inter_cluster_distance': 0.0,
            'cluster_compactness': 0.0,
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'combined_score': 0.0
        }
    
    def rank_features_by_distinctiveness(self, 
                                       features: Dict[str, np.ndarray], 
                                       cluster_labels: np.ndarray) -> List[Tuple[str, float]]:
        """
        Rank features by their distinctiveness score.
        
        Args:
            features: Dictionary of feature names to feature values
            cluster_labels: Cluster labels for each sample
            
        Returns:
            List of (feature_name, distinctiveness_score) tuples, sorted by score
        """
        distinctiveness_metrics = self.calculate_feature_distinctiveness(features, cluster_labels)
        
        # Extract combined scores
        feature_scores = []
        for feature_name, metrics in distinctiveness_metrics.items():
            score = metrics.get('combined_score', 0.0)
            feature_scores.append((feature_name, score))
        
        # Sort by score (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores
    
    def get_top_distinctive_features(self, 
                                   features: Dict[str, np.ndarray], 
                                   cluster_labels: np.ndarray, 
                                   n_features: int) -> Dict[str, np.ndarray]:
        """
        Get top N most distinctive features.
        
        Args:
            features: Dictionary of feature names to feature values
            cluster_labels: Cluster labels for each sample
            n_features: Number of top features to return
            
        Returns:
            Dictionary of top distinctive features
        """
        ranked_features = self.rank_features_by_distinctiveness(features, cluster_labels)
        
        # Select top N features
        top_features = {}
        for feature_name, _ in ranked_features[:n_features]:
            if feature_name in features:
                top_features[feature_name] = features[feature_name]
        
        return top_features


# Convenience functions
def calculate_cluster_distinctiveness(features: Dict[str, np.ndarray], 
                                    cluster_labels: np.ndarray,
                                    config: Optional[ClusterDistinctivenessConfig] = None) -> Dict[str, Dict[str, float]]:
    """Calculate cluster distinctiveness metrics for features."""
    calculator = ClusterDistinctivenessCalculator(config)
    return calculator.calculate_feature_distinctiveness(features, cluster_labels)


def rank_features_by_cluster_distinctiveness(features: Dict[str, np.ndarray], 
                                           cluster_labels: np.ndarray,
                                           config: Optional[ClusterDistinctivenessConfig] = None) -> List[Tuple[str, float]]:
    """Rank features by cluster distinctiveness."""
    calculator = ClusterDistinctivenessCalculator(config)
    return calculator.rank_features_by_distinctiveness(features, cluster_labels)


def get_top_distinctive_features(features: Dict[str, np.ndarray], 
                               cluster_labels: np.ndarray, 
                               n_features: int,
                               config: Optional[ClusterDistinctivenessConfig] = None) -> Dict[str, np.ndarray]:
    """Get top N most distinctive features."""
    calculator = ClusterDistinctivenessCalculator(config)
    return calculator.get_top_distinctive_features(features, cluster_labels, n_features)


__all__ = [
    'ClusterDistinctivenessCalculator',
    'ClusterDistinctivenessConfig',
    'calculate_cluster_distinctiveness',
    'rank_features_by_cluster_distinctiveness',
    'get_top_distinctive_features'
]