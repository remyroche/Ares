"""
Unified Clustering Algorithms for Hybrid NAS-TAS Regime System

This module provides unified clustering algorithms that combine the best approaches
from both TAS and NAS systems, including economic-aware clustering and advanced
ensemble methods.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import utility modules
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_float, safe_int,
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, get_dataframe_info, create_summary_statistics,
    optimize_dataframe_dtypes, safe_merge_dataframes, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    safe_filter_dataframe, create_data_quality_report, safe_to_parquet,
    safe_read_parquet, get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory
)

from src.utils.math_validation import (
    validate_numeric_array, safe_correlation, safe_covariance, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidation
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import M1 hardware optimizers
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available, optimize_dataframe_for_m1
    )
    M1_GPU_AVAILABLE = True
except ImportError:
    M1_GPU_AVAILABLE = False
    def get_m1_gpu_manager(): return None
    def is_m1_available(): return False
    def is_mps_available(): return False
    def optimize_dataframe_for_m1(df): return df

try:
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory
    )
    M1_MEMORY_AVAILABLE = True
except ImportError:
    M1_MEMORY_AVAILABLE = False
    def get_m1_memory_optimizer(): return None
    def optimize_dataframe_memory(df): return df

try:
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool
    )
    M1_CPU_AVAILABLE = True
except ImportError:
    M1_CPU_AVAILABLE = False
    def get_m1_cpu_optimizer(): return None
    def parallel_map_m1(func, items, max_workers=None): return [func(item) for item in items]
    def create_m1_optimized_thread_pool(max_workers=None): return None

warnings.filterwarnings('ignore')

# Initialize math validation
math_validator = MathValidation()


class ClusteringAlgorithmType(Enum):
    """Types of clustering algorithms available."""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    ECONOMIC_KMEANS = "economic_kmeans"
    ECONOMIC_HIERARCHICAL = "economic_hierarchical"
    ECONOMIC_GMM = "economic_gmm"
    ENSEMBLE_CLUSTERING = "ensemble_clustering"
    ADAPTIVE_CLUSTERING = "adaptive_clustering"


@dataclass
class ClusteringResult:
    """Result from clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    probabilities: np.ndarray
    quality_metrics: Dict[str, float]
    algorithm_used: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class UnifiedClusteringAlgorithm:
    """Unified clustering algorithm that combines multiple approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified clustering algorithm.
        
        Args:
            config: Configuration dictionary
        """
        try:
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # Validate configuration
            self._validate_config(config)
            
            # Clustering parameters with validation
            self.n_regimes = validate_positive(
                config.get('n_regimes', 8), 
                name="n_regimes"
            )
            self.algorithm_type = config.get('algorithm_type', 'adaptive_clustering')
            self.enable_economic_clustering = config.get('enable_economic_clustering', True)
            self.enable_ensemble_clustering = config.get('enable_ensemble_clustering', False)
            
            # Economic clustering parameters with validation
            self.economic_weight = validate_range(
                config.get('economic_weight', 0.3), 
                min_val=0.0, max_val=1.0, name="economic_weight"
            )
            self.momentum_weight = validate_range(
                config.get('momentum_weight', 0.25), 
                min_val=0.0, max_val=1.0, name="momentum_weight"
            )
            self.volume_weight = validate_range(
                config.get('volume_weight', 0.25), 
                min_val=0.0, max_val=1.0, name="volume_weight"
            )
            
            # Validate weight sum
            total_weight = self.economic_weight + self.momentum_weight + self.volume_weight
            if total_weight > 1.0:
                tprint_warning(f"⚠️ Total economic weights ({total_weight:.2f}) exceed 1.0, normalizing...")
                self.economic_weight /= total_weight
                self.momentum_weight /= total_weight
                self.volume_weight /= total_weight
            
            # Initialize M1 optimizers
            self._initialize_m1_optimizers()
            
            # Initialize serializers
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            
            tprint_success("✅ Unified Clustering Algorithm initialized")
            tprint_info(f"   Algorithm type: {self.algorithm_type}")
            tprint_info(f"   Economic clustering: {self.enable_economic_clustering}")
            tprint_info(f"   Ensemble clustering: {self.enable_ensemble_clustering}")
            tprint_info(f"   M1 GPU available: {M1_GPU_AVAILABLE}")
            tprint_info(f"   M1 Memory available: {M1_MEMORY_AVAILABLE}")
            tprint_info(f"   M1 CPU available: {M1_CPU_AVAILABLE}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Unified Clustering Algorithm: {e}")
            raise ValueError(f"Initialization failed: {e}") from e
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        try:
            required_keys = ['n_regimes']
            for key in required_keys:
                if key not in config:
                    tprint_warning(f"⚠️ Missing required config key: {key}")
            
            # Validate algorithm type
            valid_algorithms = [
                'kmeans', 'gaussian_mixture', 'hierarchical', 'dbscan',
                'economic_kmeans', 'economic_hierarchical', 'economic_gmm',
                'ensemble_clustering', 'adaptive_clustering'
            ]
            if config.get('algorithm_type') not in valid_algorithms:
                tprint_warning(f"⚠️ Invalid algorithm type: {config.get('algorithm_type')}")
                config['algorithm_type'] = 'adaptive_clustering'
                
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _initialize_m1_optimizers(self) -> None:
        """Initialize M1 hardware optimizers."""
        try:
            # Initialize M1 GPU manager
            if M1_GPU_AVAILABLE:
                self.m1_gpu_manager = get_m1_gpu_manager()
                if self.m1_gpu_manager and is_m1_available():
                    tprint_info("🧠 M1 GPU manager initialized")
                else:
                    tprint_warning("⚠️ M1 GPU not available")
            else:
                self.m1_gpu_manager = None
                tprint_warning("⚠️ M1 GPU utilities not available")
            
            # Initialize M1 memory optimizer
            if M1_MEMORY_AVAILABLE:
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                if self.m1_memory_optimizer:
                    tprint_info("🧠 M1 memory optimizer initialized")
                else:
                    tprint_warning("⚠️ M1 memory optimizer not available")
            else:
                self.m1_memory_optimizer = None
                tprint_warning("⚠️ M1 memory utilities not available")
            
            # Initialize M1 CPU optimizer
            if M1_CPU_AVAILABLE:
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                if self.m1_cpu_optimizer:
                    tprint_info("🧠 M1 CPU optimizer initialized")
                else:
                    tprint_warning("⚠️ M1 CPU optimizer not available")
            else:
                self.m1_cpu_optimizer = None
                tprint_warning("⚠️ M1 CPU utilities not available")
                
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimizer initialization failed: {e}")
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
    
    def cluster_features(self,
                        features: np.ndarray,
                        market_data: Optional[pd.DataFrame] = None,
                        economic_weights: Optional[np.ndarray] = None) -> ClusteringResult:
        """Perform clustering on features.
        
        Args:
            features: Feature matrix
            market_data: Optional market data for economic clustering
            economic_weights: Optional economic weights for features
            
        Returns:
            ClusteringResult with clustering results
        """
        start_time = time.time()
        
        try:
            tprint_info("🔍 Starting unified clustering...")
            
            # Validate inputs
            self._validate_clustering_inputs(features, market_data, economic_weights)
            
            # Optimize features for M1 if available
            features = self._optimize_features_for_m1(features)
            
            # Choose clustering strategy
            with tprint_timer("Clustering execution"):
                if self.algorithm_type == 'adaptive_clustering':
                    result = self._adaptive_clustering(features, market_data, economic_weights)
                elif self.algorithm_type == 'ensemble_clustering':
                    result = self._ensemble_clustering(features, market_data, economic_weights)
                elif self.algorithm_type.startswith('economic_'):
                    result = self._economic_clustering(features, market_data, economic_weights)
                else:
                    result = self._standard_clustering(features, economic_weights)
            
            execution_time = time.time() - start_time
            
            # Log performance metrics
            tprint_performance("Unified clustering", execution_time)
            tprint_success(f"✅ Clustering completed successfully in {execution_time:.3f}s")
            
            return ClusteringResult(
                labels=result['labels'],
                cluster_centers=result['cluster_centers'],
                probabilities=result['probabilities'],
                quality_metrics=result['quality_metrics'],
                algorithm_used=self.algorithm_type,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Unified clustering failed: {e}")
            tprint_debug(f"   Error type: {type(e).__name__}")
            tprint_debug(f"   Execution time: {execution_time:.3f}s")
            
            return ClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                probabilities=np.array([]),
                quality_metrics={},
                algorithm_used=self.algorithm_type,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _validate_clustering_inputs(self, 
                                  features: np.ndarray, 
                                  market_data: Optional[pd.DataFrame], 
                                  economic_weights: Optional[np.ndarray]) -> None:
        """Validate clustering inputs."""
        try:
            # Validate features
            if features is None:
                raise ValueError("Features cannot be None")
            
            if not isinstance(features, np.ndarray):
                raise TypeError(f"Features must be numpy array, got {type(features)}")
            
            if features.size == 0:
                raise ValueError("Features cannot be empty")
            
            # Validate numeric array
            features = validate_numeric_array(features, "features")
            
            # Validate market data if provided
            if market_data is not None:
                if not isinstance(market_data, pd.DataFrame):
                    raise TypeError(f"Market data must be DataFrame, got {type(market_data)}")
                
                if market_data.empty:
                    raise ValueError("Market data cannot be empty")
                
                # Validate required columns
                required_columns = ['close']
                if not validate_dataframe_columns(market_data, required_columns):
                    raise ValueError(f"Market data missing required columns: {required_columns}")
            
            # Validate economic weights if provided
            if economic_weights is not None:
                if not isinstance(economic_weights, np.ndarray):
                    raise TypeError(f"Economic weights must be numpy array, got {type(economic_weights)}")
                
                if len(economic_weights) != features.shape[1]:
                    raise ValueError(f"Economic weights length ({len(economic_weights)}) must match features columns ({features.shape[1]})")
                
                economic_weights = validate_numeric_array(economic_weights, "economic_weights")
            
            tprint_debug("✅ Input validation passed")
            
        except Exception as e:
            tprint_error(f"❌ Input validation failed: {e}")
            raise
    
    def _optimize_features_for_m1(self, features: np.ndarray) -> np.ndarray:
        """Optimize features for M1 hardware."""
        try:
            if not M1_GPU_AVAILABLE or not is_m1_available():
                return features
            
            # Use M1 GPU optimization if available
            if self.m1_gpu_manager:
                with gpu_context("clustering_features"):
                    optimized_features = self.m1_gpu_manager.optimize_tensor_operations(features)
                    if optimized_features is not None:
                        tprint_debug("🧠 Features optimized for M1 GPU")
                        return optimized_features
            
            # Use M1 memory optimization
            if self.m1_memory_optimizer:
                with memory_checkpoint("clustering_features"):
                    # Convert to DataFrame for memory optimization
                    df = pd.DataFrame(features)
                    optimized_df = self.m1_memory_optimizer.optimize_dataframe_memory(df)
                    optimized_features = optimized_df.values
                    tprint_debug("🧠 Features optimized for M1 memory")
                    return optimized_features
            
            return features
            
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimization failed: {e}")
            return features
    
    def _adaptive_clustering(self,
                           features: np.ndarray,
                           market_data: Optional[pd.DataFrame],
                           economic_weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Adaptive clustering that selects best algorithm."""
        try:
            tprint_info("🔍 Performing adaptive clustering...")
            
            best_score = -1
            best_result = None
            best_algorithm = None
            algorithm_scores = {}
            
            # Try different algorithms
            algorithms_to_try = ['kmeans', 'gaussian_mixture', 'hierarchical']
            
            if self.enable_economic_clustering and market_data is not None:
                algorithms_to_try.extend(['economic_kmeans', 'economic_gmm'])
                tprint_debug("   Economic clustering enabled")
            
            tprint_debug(f"   Testing {len(algorithms_to_try)} algorithms: {algorithms_to_try}")
            
            for i, algorithm in enumerate(algorithms_to_try):
                try:
                    tprint_progress(i + 1, len(algorithms_to_try), f"Testing {algorithm}")
                    
                    if algorithm.startswith('economic_'):
                        result = self._economic_clustering(features, market_data, economic_weights, algorithm)
                    else:
                        result = self._standard_clustering(features, economic_weights, algorithm)
                    
                    # Calculate quality score
                    score = self._calculate_quality_score(features, result['labels'])
                    algorithm_scores[algorithm] = score
                    
                    tprint_debug(f"   {algorithm}: score = {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_algorithm = algorithm
                        tprint_debug(f"   New best: {algorithm} (score: {score:.3f})")
                
                except Exception as e:
                    tprint_warning(f"⚠️ Algorithm {algorithm} failed: {e}")
                    algorithm_scores[algorithm] = -1
                    continue
            
            if best_result is None:
                raise ValueError("All clustering algorithms failed")
            
            tprint_success(f"✅ Selected algorithm: {best_algorithm} (score: {best_score:.3f})")
            tprint_debug(f"   All scores: {algorithm_scores}")
            
            return best_result
            
        except Exception as e:
            tprint_error(f"❌ Adaptive clustering failed: {e}")
            tprint_warning("⚠️ Falling back to K-means clustering")
            # Fallback to kmeans
            return self._standard_clustering(features, economic_weights, 'kmeans')
    
    def _ensemble_clustering(self,
                           features: np.ndarray,
                           market_data: Optional[pd.DataFrame],
                           economic_weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Ensemble clustering combining multiple algorithms."""
        try:
            tprint_info("🔍 Performing ensemble clustering...")
            
            # Get predictions from multiple algorithms
            predictions = []
            algorithms = ['kmeans', 'gaussian_mixture', 'hierarchical']
            algorithm_results = {}
            
            tprint_debug(f"   Running {len(algorithms)} algorithms for ensemble")
            
            for i, algorithm in enumerate(algorithms):
                try:
                    tprint_progress(i + 1, len(algorithms), f"Running {algorithm}")
                    result = self._standard_clustering(features, economic_weights, algorithm)
                    predictions.append(result['labels'])
                    algorithm_results[algorithm] = result
                    tprint_debug(f"   {algorithm}: {len(result['labels'])} labels generated")
                except Exception as e:
                    tprint_warning(f"⚠️ Algorithm {algorithm} failed: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No algorithms succeeded")
            
            tprint_info(f"   Successfully ran {len(predictions)} algorithms")
            
            # Combine predictions using voting
            n_samples = len(features)
            votes = np.zeros((n_samples, self.n_regimes))
            
            for pred in predictions:
                for i, label in enumerate(pred):
                    if 0 <= label < self.n_regimes:
                        votes[i, label] += 1
            
            # Final labels based on majority vote
            final_labels = np.argmax(votes, axis=1)
            
            # Use KMeans on final predictions for refinement
            tprint_debug("   Refining ensemble results with K-means")
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            refined_labels = kmeans.fit_predict(votes)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(features, refined_labels)
            
            tprint_success(f"✅ Ensemble clustering completed with {len(predictions)} algorithms")
            
            return {
                'labels': refined_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'probabilities': votes / len(predictions),
                'quality_metrics': quality_metrics,
                'ensemble_info': {
                    'algorithms_used': list(algorithm_results.keys()),
                    'n_algorithms': len(predictions),
                    'voting_matrix_shape': votes.shape
                }
            }
            
        except Exception as e:
            tprint_error(f"❌ Ensemble clustering failed: {e}")
            tprint_warning("⚠️ Falling back to K-means clustering")
            return self._standard_clustering(features, economic_weights, 'kmeans')
    
    def _economic_clustering(self,
                           features: np.ndarray,
                           market_data: pd.DataFrame,
                           economic_weights: Optional[np.ndarray],
                           algorithm: str = 'economic_kmeans') -> Dict[str, Any]:
        """Economic-aware clustering."""
        try:
            tprint_info(f"🔍 Performing {algorithm} clustering...")
            
            # Extract economic features
            tprint_debug("   Extracting economic features")
            economic_features = self._extract_economic_features(features, market_data)
            
            # Calculate momentum features
            tprint_debug("   Calculating momentum features")
            momentum_features = self._calculate_momentum_features(market_data)
            
            # Calculate volume features
            tprint_debug("   Calculating volume features")
            volume_features = self._calculate_volume_features(market_data)
            
            # Combine all features with economic weighting
            tprint_debug("   Combining economic features")
            combined_features = self._combine_economic_features(
                features, economic_features, momentum_features, volume_features
            )
            
            # Apply economic weights if provided
            if economic_weights is not None:
                tprint_debug("   Applying economic weights")
                combined_features = self._apply_economic_weights(combined_features, economic_weights)
            
            # Perform clustering
            tprint_debug(f"   Running {algorithm} on combined features")
            if algorithm == 'economic_kmeans':
                result = self._economic_kmeans(combined_features)
            elif algorithm == 'economic_gmm':
                result = self._economic_gmm(combined_features)
            elif algorithm == 'economic_hierarchical':
                result = self._economic_hierarchical(combined_features)
            else:
                result = self._economic_kmeans(combined_features)
            
            # Add economic clustering metadata
            result['economic_info'] = {
                'economic_features_shape': economic_features.shape,
                'momentum_features_shape': momentum_features.shape,
                'volume_features_shape': volume_features.shape,
                'combined_features_shape': combined_features.shape,
                'economic_weights_applied': economic_weights is not None
            }
            
            tprint_success(f"✅ {algorithm} clustering completed")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Economic clustering failed: {e}")
            tprint_warning("⚠️ Falling back to standard K-means clustering")
            return self._standard_clustering(features, economic_weights, 'kmeans')
    
    def _standard_clustering(self,
                           features: np.ndarray,
                           economic_weights: Optional[np.ndarray],
                           algorithm: str = 'kmeans') -> Dict[str, Any]:
        """Standard clustering algorithms."""
        try:
            self.logger.info(f"🔍 Performing {algorithm} clustering...")
            
            # Apply weights if provided
            if economic_weights is not None:
                features = self._apply_economic_weights(features, economic_weights)
            
            if algorithm == 'kmeans':
                return self._kmeans_clustering(features)
            elif algorithm == 'gaussian_mixture':
                return self._gmm_clustering(features)
            elif algorithm == 'hierarchical':
                return self._hierarchical_clustering(features)
            else:
                return self._kmeans_clustering(features)
            
        except Exception as e:
            self.logger.error(f"Standard clustering failed: {e}")
            raise
    
    def _kmeans_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """K-means clustering."""
        try:
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            probabilities = self._calculate_cluster_probabilities(features, labels, kmeans)
            quality_metrics = self._calculate_quality_metrics(features, labels)
            
            return {
                'labels': labels,
                'cluster_centers': kmeans.cluster_centers_,
                'probabilities': probabilities,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"K-means clustering failed: {e}")
            raise
    
    def _gmm_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Gaussian Mixture Model clustering."""
        try:
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42, n_init=5)
            labels = gmm.fit_predict(features)
            probabilities = gmm.predict_proba(features)
            quality_metrics = self._calculate_quality_metrics(features, labels)
            
            return {
                'labels': labels,
                'cluster_centers': gmm.means_,
                'probabilities': probabilities,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"GMM clustering failed: {e}")
            raise
    
    def _hierarchical_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Hierarchical clustering."""
        try:
            hierarchical = AgglomerativeClustering(n_clusters=self.n_regimes, linkage='ward')
            labels = hierarchical.fit_predict(features)
            probabilities = self._calculate_cluster_probabilities(features, labels)
            quality_metrics = self._calculate_quality_metrics(features, labels)
            
            return {
                'labels': labels,
                'cluster_centers': np.array([]),  # Hierarchical doesn't provide centers
                'probabilities': probabilities,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Hierarchical clustering failed: {e}")
            raise
    
    def _economic_kmeans(self, features: np.ndarray) -> Dict[str, Any]:
        """Economic-aware K-means clustering."""
        try:
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            probabilities = self._calculate_cluster_probabilities(features, labels, kmeans)
            quality_metrics = self._calculate_quality_metrics(features, labels)
            
            return {
                'labels': labels,
                'cluster_centers': kmeans.cluster_centers_,
                'probabilities': probabilities,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Economic K-means failed: {e}")
            raise
    
    def _economic_gmm(self, features: np.ndarray) -> Dict[str, Any]:
        """Economic-aware Gaussian Mixture Model clustering."""
        try:
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42, n_init=5)
            labels = gmm.fit_predict(features)
            probabilities = gmm.predict_proba(features)
            quality_metrics = self._calculate_quality_metrics(features, labels)
            
            return {
                'labels': labels,
                'cluster_centers': gmm.means_,
                'probabilities': probabilities,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Economic GMM failed: {e}")
            raise
    
    def _economic_hierarchical(self, features: np.ndarray) -> Dict[str, Any]:
        """Economic-aware hierarchical clustering."""
        try:
            hierarchical = AgglomerativeClustering(n_clusters=self.n_regimes, linkage='ward')
            labels = hierarchical.fit_predict(features)
            probabilities = self._calculate_cluster_probabilities(features, labels)
            quality_metrics = self._calculate_quality_metrics(features, labels)
            
            return {
                'labels': labels,
                'cluster_centers': np.array([]),  # Hierarchical doesn't provide centers
                'probabilities': probabilities,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Economic hierarchical failed: {e}")
            raise
    
    def _extract_economic_features(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Extract economic features from market data."""
        try:
            economic_features_list = []
            
            # Price-based economic features
            close_prices = market_data['close'].values
            
            # Volatility features
            returns = np.diff(close_prices, prepend=close_prices[0])
            volatility_features = self._calculate_volatility_features(returns)
            economic_features_list.append(volatility_features)
            
            # Trend features
            trend_features = self._calculate_trend_features(close_prices)
            economic_features_list.append(trend_features)
            
            # Combine all economic features
            if economic_features_list:
                return np.hstack([f.reshape(-1, 1) if f.ndim == 1 else f for f in economic_features_list])
            else:
                return np.zeros((len(market_data), 1))
            
        except Exception as e:
            self.logger.warning(f"Economic feature extraction failed: {e}")
            return np.zeros((len(market_data), 1))
    
    def _calculate_momentum_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate momentum features."""
        try:
            close_prices = market_data['close'].values
            momentum_features_list = []
            
            # Price momentum for different periods
            for period in [1, 2, 5, 10]:
                if len(close_prices) > period:
                    momentum = (close_prices - np.roll(close_prices, period)) / (np.roll(close_prices, period) + 1e-8)
                    momentum_features_list.append(momentum.reshape(-1, 1))
            
            if momentum_features_list:
                return np.hstack(momentum_features_list)
            else:
                return np.zeros((len(market_data), 1))
            
        except Exception as e:
            self.logger.warning(f"Momentum feature calculation failed: {e}")
            return np.zeros((len(market_data), 1))
    
    def _calculate_volume_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate volume-based features."""
        try:
            volume_features_list = []
            
            if 'volume' in market_data.columns:
                volume = market_data['volume'].values
                
                # Volume features
                volume_ma = pd.Series(volume).rolling(window=20, min_periods=5).mean().fillna(method='bfill').values
                volume_std = pd.Series(volume).rolling(window=20, min_periods=5).std().fillna(method='bfill').values
                
                volume_features_list.append(volume_ma.reshape(-1, 1))
                volume_features_list.append(volume_std.reshape(-1, 1))
            
            if volume_features_list:
                return np.hstack(volume_features_list)
            else:
                return np.zeros((len(market_data), 1))
            
        except Exception as e:
            self.logger.warning(f"Volume feature calculation failed: {e}")
            return np.zeros((len(market_data), 1))
    
    def _combine_economic_features(self,
                                 base_features: np.ndarray,
                                 economic_features: np.ndarray,
                                 momentum_features: np.ndarray,
                                 volume_features: np.ndarray) -> np.ndarray:
        """Combine features with economic weighting."""
        try:
            features_list = []
            
            # Add base features with reduced weight
            if base_features.size > 0:
                base_weighted = base_features * (1 - self.economic_weight - self.momentum_weight - self.volume_weight)
                features_list.append(base_weighted)
            
            # Add economic features
            if economic_features.size > 0:
                economic_weighted = economic_features * self.economic_weight
                features_list.append(economic_weighted)
            
            # Add momentum features
            if momentum_features.size > 0:
                momentum_weighted = momentum_features * self.momentum_weight
                features_list.append(momentum_weighted)
            
            # Add volume features
            if volume_features.size > 0:
                volume_weighted = volume_features * self.volume_weight
                features_list.append(volume_weighted)
            
            if features_list:
                return np.hstack(features_list)
            else:
                return np.zeros((base_features.shape[0], 1))
            
        except Exception as e:
            self.logger.warning(f"Feature combination failed: {e}")
            return base_features
    
    def _apply_economic_weights(self, features: np.ndarray, economic_weights: np.ndarray) -> np.ndarray:
        """Apply economic weights to features."""
        try:
            if len(economic_weights) == features.shape[1]:
                return features * economic_weights.reshape(1, -1)
            else:
                return features
        except Exception as e:
            self.logger.warning(f"Economic weight application failed: {e}")
            return features
    
    def _calculate_volatility_features(self, returns: np.ndarray) -> np.ndarray:
        """Calculate volatility-based features."""
        try:
            features_list = []
            
            # Rolling volatility
            for window in [5, 10, 20, 50]:
                if len(returns) > window:
                    rolling_vol = pd.Series(np.abs(returns)).rolling(window=window, min_periods=window//2).std()
                    rolling_vol = rolling_vol.fillna(rolling_vol.mean()).values
                    features_list.append(rolling_vol.reshape(-1, 1))
            
            return np.hstack(features_list) if features_list else np.zeros((len(returns), 1))
            
        except Exception as e:
            self.logger.warning(f"Volatility feature calculation failed: {e}")
            return np.zeros((len(returns), 1))
    
    def _calculate_trend_features(self, prices: np.ndarray) -> np.ndarray:
        """Calculate trend-based features."""
        try:
            features_list = []
            
            # Trend strength for different periods
            for period in [5, 10, 20, 50]:
                if len(prices) > period:
                    # Simple trend calculation
                    trend = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                    features_list.append(np.array([trend]))
            
            return np.hstack(features_list) if features_list else np.zeros((len(prices), 1))
            
        except Exception as e:
            self.logger.warning(f"Trend feature calculation failed: {e}")
            return np.zeros((len(prices), 1))
    
    def _calculate_cluster_probabilities(self, 
                                       features: np.ndarray, 
                                       labels: np.ndarray, 
                                       clusterer=None) -> np.ndarray:
        """Calculate cluster membership probabilities."""
        try:
            n_samples = len(features)
            n_clusters = len(set(labels))
            
            if n_clusters == 0:
                return np.ones((n_samples, 1)) * 0.5
            
            # For K-means, estimate probabilities based on distance to centroids
            if clusterer is not None and hasattr(clusterer, 'cluster_centers_'):
                centroids = clusterer.cluster_centers_
                probabilities = np.zeros((n_samples, n_clusters))
                
                for i, label in enumerate(labels):
                    if 0 <= label < n_clusters:
                        # Distance to assigned cluster
                        assigned_distance = np.linalg.norm(features[i] - centroids[label])
                        
                        # Distances to all clusters
                        distances = np.linalg.norm(features[i] - centroids, axis=1)
                        
                        # Convert distances to probabilities (closer = higher probability)
                        if np.min(distances) > 0:
                            probabilities[i] = 1 / (distances + 1e-8)
                            probabilities[i] /= np.sum(probabilities[i])
                        else:
                            probabilities[i, label] = 1.0
                    else:
                        probabilities[i] = 1.0 / n_clusters
                
                return probabilities
            else:
                # Uniform probabilities for other algorithms
                probabilities = np.ones((n_samples, n_clusters)) / n_clusters
                for i, label in enumerate(labels):
                    if 0 <= label < n_clusters:
                        probabilities[i] *= 0.7  # Higher probability for assigned cluster
                        probabilities[i, label] += 0.3  # Boost assigned cluster
                        probabilities[i] /= np.sum(probabilities[i])  # Renormalize
                
                return probabilities
            
        except Exception as e:
            self.logger.warning(f"Probability calculation failed: {e}")
            n_clusters = len(set(labels))
            return np.ones((len(features), n_clusters)) / n_clusters
    
    def _calculate_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        try:
            metrics = {}
            
            unique_labels = set(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                tprint_warning("⚠️ Insufficient clusters for quality metrics")
                return {'error': 'Insufficient clusters', 'n_clusters': n_clusters}
            
            tprint_debug(f"   Calculating quality metrics for {n_clusters} clusters")
            
            # Standard clustering metrics with safe calculations
            try:
                metrics['silhouette_score'] = math_safe(
                    silhouette_score, features, labels, default=0.0
                )
            except Exception as e:
                tprint_warning(f"⚠️ Silhouette score calculation failed: {e}")
                metrics['silhouette_score'] = 0.0
            
            try:
                metrics['calinski_harabasz_score'] = math_safe(
                    calinski_harabasz_score, features, labels, default=0.0
                )
            except Exception as e:
                tprint_warning(f"⚠️ Calinski-Harabasz score calculation failed: {e}")
                metrics['calinski_harabasz_score'] = 0.0
            
            try:
                metrics['davies_bouldin_score'] = math_safe(
                    davies_bouldin_score, features, labels, default=0.0
                )
            except Exception as e:
                tprint_warning(f"⚠️ Davies-Bouldin score calculation failed: {e}")
                metrics['davies_bouldin_score'] = 0.0
            
            # Regime-specific metrics
            regime_sizes = np.bincount(labels, minlength=n_clusters)
            if np.mean(regime_sizes) > 0:
                metrics['regime_balance'] = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes))
            else:
                metrics['regime_balance'] = 0.0
            
            metrics['min_regime_size'] = int(np.min(regime_sizes))
            metrics['max_regime_size'] = int(np.max(regime_sizes))
            metrics['total_samples'] = len(labels)
            metrics['n_clusters'] = n_clusters
            
            # Additional quality metrics
            metrics['cluster_compactness'] = self._calculate_cluster_compactness(features, labels)
            metrics['cluster_separation'] = self._calculate_cluster_separation(features, labels)
            
            tprint_debug(f"   Quality metrics: silhouette={metrics['silhouette_score']:.3f}, "
                        f"calinski_harabasz={metrics['calinski_harabasz_score']:.3f}")
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Quality metrics calculation failed: {e}")
            return {'error': str(e), 'n_clusters': len(set(labels))}
    
    def _calculate_cluster_compactness(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster compactness metric."""
        try:
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            compactness = 0.0
            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    compactness += np.mean(distances)
            
            return compactness / len(unique_labels)
        except Exception:
            return 0.0
    
    def _calculate_cluster_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster separation metric."""
        try:
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            centroids = []
            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 0:
                    centroids.append(np.mean(cluster_points, axis=0))
            
            if len(centroids) < 2:
                return 0.0
            
            centroids = np.array(centroids)
            min_distance = float('inf')
            
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    min_distance = min(min_distance, distance)
            
            return min_distance if min_distance != float('inf') else 0.0
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate overall quality score for algorithm selection."""
        try:
            if len(set(labels)) < 2:
                return 0.0
            
            # Get quality metrics
            metrics = self._calculate_quality_metrics(features, labels)
            
            # Combine scores (normalize to 0-1 range)
            silhouette = metrics.get('silhouette_score', 0.0)
            ch_score = metrics.get('calinski_harabasz_score', 0.0)
            regime_balance = metrics.get('regime_balance', 0.0)
            
            # Normalize scores
            normalized_silhouette = max(0, min(1, silhouette))
            normalized_ch = min(ch_score / 1000, 1.0)
            
            # Combined score
            score = 0.4 * normalized_silhouette + 0.3 * normalized_ch + 0.3 * regime_balance
            return score
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0


    def save_results(self, result: ClusteringResult, filepath: str, format: str = 'json') -> bool:
        """Save clustering results to file.
        
        Args:
            result: ClusteringResult to save
            filepath: Path to save the results
            format: Format to save ('json', 'pickle', 'parquet')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving clustering results to {filepath}")
            
            # Prepare data for serialization
            data = {
                'labels': result.labels.tolist() if hasattr(result.labels, 'tolist') else result.labels,
                'cluster_centers': result.cluster_centers.tolist() if hasattr(result.cluster_centers, 'tolist') else result.cluster_centers,
                'probabilities': result.probabilities.tolist() if hasattr(result.probabilities, 'tolist') else result.probabilities,
                'quality_metrics': result.quality_metrics,
                'algorithm_used': result.algorithm_used,
                'execution_time': result.execution_time,
                'success': result.success,
                'error_message': result.error_message,
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
            
            if format == 'json':
                success = self.json_serializer.save(data, filepath)
            elif format == 'pickle':
                success = self.pickle_serializer.save(data, filepath)
            elif format == 'parquet':
                # Convert to DataFrame for parquet
                df = pd.DataFrame({
                    'labels': result.labels,
                    'cluster_centers': [result.cluster_centers.tolist()] * len(result.labels),
                    'probabilities': [result.probabilities.tolist()] * len(result.labels)
                })
                success = self.parquet_serializer.save(df, filepath)
            else:
                success = self.universal_serializer.save(data, filepath, format)
            
            if success:
                tprint_success(f"✅ Results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save results to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[ClusteringResult]:
        """Load clustering results from file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            ClusteringResult if successful, None otherwise
        """
        try:
            tprint_info(f"📂 Loading clustering results from {filepath}")
            
            # Determine format from file extension
            if filepath.endswith('.json'):
                data = self.json_serializer.load(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                data = self.pickle_serializer.load(filepath)
            elif filepath.endswith('.parquet'):
                df = self.parquet_serializer.load(filepath)
                if df is not None:
                    data = {
                        'labels': df['labels'].values,
                        'cluster_centers': df['cluster_centers'].iloc[0],
                        'probabilities': df['probabilities'].iloc[0],
                        'quality_metrics': {},
                        'algorithm_used': 'unknown',
                        'execution_time': 0.0,
                        'success': True,
                        'error_message': None
                    }
                else:
                    data = None
            else:
                data = self.universal_serializer.load(filepath)
            
            if data is None:
                tprint_error(f"❌ Failed to load data from {filepath}")
                return None
            
            # Convert back to ClusteringResult
            result = ClusteringResult(
                labels=np.array(data['labels']),
                cluster_centers=np.array(data['cluster_centers']),
                probabilities=np.array(data['probabilities']),
                quality_metrics=data['quality_metrics'],
                algorithm_used=data['algorithm_used'],
                execution_time=data['execution_time'],
                success=data['success'],
                error_message=data.get('error_message')
            )
            
            tprint_success(f"✅ Results loaded successfully from {filepath}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error loading results: {e}")
            return None
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the clustering algorithm.
        
        Returns:
            Dictionary with algorithm information
        """
        return {
            'algorithm_type': self.algorithm_type,
            'n_regimes': self.n_regimes,
            'enable_economic_clustering': self.enable_economic_clustering,
            'enable_ensemble_clustering': self.enable_ensemble_clustering,
            'economic_weight': self.economic_weight,
            'momentum_weight': self.momentum_weight,
            'volume_weight': self.volume_weight,
            'm1_gpu_available': M1_GPU_AVAILABLE and self.m1_gpu_manager is not None,
            'm1_memory_available': M1_MEMORY_AVAILABLE and self.m1_memory_optimizer is not None,
            'm1_cpu_available': M1_CPU_AVAILABLE and self.m1_cpu_optimizer is not None,
            'config': self.config
        }


def create_unified_clustering_algorithm(config: Dict[str, Any]) -> UnifiedClusteringAlgorithm:
    """Create a unified clustering algorithm instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UnifiedClusteringAlgorithm instance
    """
    try:
        tprint_info("🏗️ Creating unified clustering algorithm")
        algorithm = UnifiedClusteringAlgorithm(config)
        tprint_success("✅ Unified clustering algorithm created successfully")
        return algorithm
    except Exception as e:
        tprint_error(f"❌ Failed to create unified clustering algorithm: {e}")
        raise