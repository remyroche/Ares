"""
Unified Clustering Algorithms for Hybrid NAS-TAS Regime System

This module provides unified clustering algorithms that combine the best approaches
from both TAS and NAS systems, including economic-aware clustering and advanced
ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Import tprint utilities
from src.utils.tprint import (

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
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


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
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Clustering parameters
        self.n_regimes = config.get('n_regimes', 8)
        self.algorithm_type = config.get('algorithm_type', 'adaptive_clustering')
        self.enable_economic_clustering = config.get('enable_economic_clustering', True)
        self.enable_ensemble_clustering = config.get('enable_ensemble_clustering', False)
        
        # Economic clustering parameters
        self.economic_weight = config.get('economic_weight', 0.3)
        self.momentum_weight = config.get('momentum_weight', 0.25)
        self.volume_weight = config.get('volume_weight', 0.25)
        
        self.logger.info("✅ Unified Clustering Algorithm initialized")
        self.logger.info(f"   Algorithm type: {self.algorithm_type}")
        self.logger.info(f"   Economic clustering: {self.enable_economic_clustering}")
        self.logger.info(f"   Ensemble clustering: {self.enable_ensemble_clustering}")
    
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
        try:
            tprint("🔍 [UNIFIED_CLUSTERING] Starting unified clustering", color="blue", bold=True)
            tprint_debug(f"📊 [UNIFIED_CLUSTERING] Features shape: {features.shape}")
            tprint_debug(f"📊 [UNIFIED_CLUSTERING] Market data shape: {market_data.shape if market_data is not None else 'None'}")
            tprint_debug(f"⚙️ [UNIFIED_CLUSTERING] Algorithm type: {self.algorithm_type}")
            tprint_debug(f"⚙️ [UNIFIED_CLUSTERING] Economic clustering: {self.enable_economic_clustering}")
            tprint_debug(f"⚙️ [UNIFIED_CLUSTERING] Ensemble clustering: {self.enable_ensemble_clustering}")
            self.logger.info("🔍 Starting unified clustering...")
            start_time = time.time()
            
            # Choose clustering strategy
            tprint(f"🎯 [UNIFIED_CLUSTERING] Using clustering strategy: {self.algorithm_type}", color="magenta")
            if self.algorithm_type == 'adaptive_clustering':
                tprint("🧠 [UNIFIED_CLUSTERING] Using adaptive clustering", color="blue")
                result = self._adaptive_clustering(features, market_data, economic_weights)
            elif self.algorithm_type == 'ensemble_clustering':
                tprint("🎭 [UNIFIED_CLUSTERING] Using ensemble clustering", color="blue")
                result = self._ensemble_clustering(features, market_data, economic_weights)
            elif self.algorithm_type.startswith('economic_'):
                tprint("💰 [UNIFIED_CLUSTERING] Using economic clustering", color="blue")
                result = self._economic_clustering(features, market_data, economic_weights)
            else:
                tprint("📊 [UNIFIED_CLUSTERING] Using standard clustering", color="blue")
                result = self._standard_clustering(features, economic_weights)
            
            execution_time = time.time() - start_time
            unique_clusters = len(set(result['labels']))
            
            tprint_success(f"✅ [UNIFIED_CLUSTERING] Unified clustering completed in {execution_time:.2f}s")
            tprint_performance("Unified clustering", execution_time)
            tprint_debug(f"🔧 [UNIFIED_CLUSTERING] Algorithm used: {self.algorithm_type}")
            tprint_debug(f"📊 [UNIFIED_CLUSTERING] Number of clusters: {unique_clusters}")
            
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
            tprint_error(f"❌ [UNIFIED_CLUSTERING] Unified clustering failed: {e}")
            tprint_debug(f"🔍 [UNIFIED_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"⏱️ [UNIFIED_CLUSTERING] Execution time before failure: {execution_time:.2f}s")
            self.logger.error(f"❌ Unified clustering failed: {e}")
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
    
    def _adaptive_clustering(self,
                           features: np.ndarray,
                           market_data: Optional[pd.DataFrame],
                           economic_weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Adaptive clustering that selects best algorithm."""
        try:
            self.logger.info("🔍 Performing adaptive clustering...")
            
            best_score = -1
            best_result = None
            best_algorithm = None
            
            # Try different algorithms
            algorithms_to_try = ['kmeans', 'gaussian_mixture', 'hierarchical']
            
            if self.enable_economic_clustering and market_data is not None:
                algorithms_to_try.extend(['economic_kmeans', 'economic_gmm'])
            
            for algorithm in algorithms_to_try:
                try:
                    if algorithm.startswith('economic_'):
                        result = self._economic_clustering(features, market_data, economic_weights, algorithm)
                    else:
                        result = self._standard_clustering(features, economic_weights, algorithm)
                    
                    # Calculate quality score
                    score = self._calculate_quality_score(features, result['labels'])
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_algorithm = algorithm
                
                except Exception as e:
                    self.logger.warning(f"Algorithm {algorithm} failed: {e}")
                    continue
            
            if best_result is None:
                raise ValueError("All clustering algorithms failed")
            
            self.logger.info(f"   Selected algorithm: {best_algorithm} (score: {best_score:.3f})")
            return best_result
            
        except Exception as e:
            self.logger.error(f"Adaptive clustering failed: {e}")
            # Fallback to kmeans
            return self._standard_clustering(features, economic_weights, 'kmeans')
    
    def _ensemble_clustering(self,
                           features: np.ndarray,
                           market_data: Optional[pd.DataFrame],
                           economic_weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Ensemble clustering combining multiple algorithms."""
        try:
            self.logger.info("🔍 Performing ensemble clustering...")
            
            # Get predictions from multiple algorithms
            predictions = []
            algorithms = ['kmeans', 'gaussian_mixture', 'hierarchical']
            
            for algorithm in algorithms:
                try:
                    result = self._standard_clustering(features, economic_weights, algorithm)
                    predictions.append(result['labels'])
                except:
                    continue
            
            if not predictions:
                raise ValueError("No algorithms succeeded")
            
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
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            refined_labels = kmeans.fit_predict(votes)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(features, refined_labels)
            
            return {
                'labels': refined_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'probabilities': votes / len(predictions),
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble clustering failed: {e}")
            return self._standard_clustering(features, economic_weights, 'kmeans')
    
    def _economic_clustering(self,
                           features: np.ndarray,
                           market_data: pd.DataFrame,
                           economic_weights: Optional[np.ndarray],
                           algorithm: str = 'economic_kmeans') -> Dict[str, Any]:
        """Economic-aware clustering."""
        try:
            self.logger.info(f"🔍 Performing {algorithm} clustering...")
            
            # Extract economic features
            economic_features = self._extract_economic_features(features, market_data)
            
            # Calculate momentum features
            momentum_features = self._calculate_momentum_features(market_data)
            
            # Calculate volume features
            volume_features = self._calculate_volume_features(market_data)
            
            # Combine all features with economic weighting
            combined_features = self._combine_economic_features(
                features, economic_features, momentum_features, volume_features
            )
            
            # Apply economic weights if provided
            if economic_weights is not None:
                combined_features = self._apply_economic_weights(combined_features, economic_weights)
            
            # Perform clustering
            if algorithm == 'economic_kmeans':
                return self._economic_kmeans(combined_features)
            elif algorithm == 'economic_gmm':
                return self._economic_gmm(combined_features)
            elif algorithm == 'economic_hierarchical':
                return self._economic_hierarchical(combined_features)
            else:
                return self._economic_kmeans(combined_features)
            
        except Exception as e:
            self.logger.error(f"Economic clustering failed: {e}")
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
            n_samples = len(returns)

            # Rolling volatility
            for window in [5, 10, 20, 50]:
                if len(returns) > window:
                    rolling_vol = pd.Series(np.abs(returns)).rolling(window=window, min_periods=window//2).std()
                    rolling_vol = rolling_vol.fillna(rolling_vol.mean()).values
                    features_list.append(rolling_vol.reshape(-1, 1))
                else:
                    # Create dummy feature with same length
                    dummy_vol = np.full(n_samples, rolling_vol.mean() if 'rolling_vol' in locals() else 0.01)
                    features_list.append(dummy_vol.reshape(-1, 1))

            return np.hstack(features_list) if features_list else np.zeros((len(returns), 1))

        except Exception as e:
            self.logger.warning(f"Volatility feature calculation failed: {e}")
            return np.zeros((len(returns), 1))
    
    def _calculate_trend_features(self, prices: np.ndarray) -> np.ndarray:
        """Calculate trend-based features."""
        try:
            features_list = []
            n_samples = len(prices)

            # Trend strength for different periods
            for period in [5, 10, 20, 50]:
                if len(prices) > period:
                    # Simple trend calculation - repeat for all time periods
                    trend = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                    # Create array with same length as prices to match dimensions
                    trend_array = np.full(n_samples, trend)
                    features_list.append(trend_array.reshape(-1, 1))
                else:
                    # Create dummy trend feature
                    dummy_trend = np.full(n_samples, 0.0)
                    features_list.append(dummy_trend.reshape(-1, 1))

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
                return {'error': 'Insufficient clusters'}
            
            # Standard clustering metrics
            try:
                metrics['silhouette_score'] = silhouette_score(features, labels)
            except:
                metrics['silhouette_score'] = 0.0
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            except:
                metrics['calinski_harabasz_score'] = 0.0
            
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
            except:
                metrics['davies_bouldin_score'] = 0.0
            
            # Regime-specific metrics
            regime_sizes = np.bincount(labels, minlength=n_clusters)
            metrics['regime_balance'] = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes))
            metrics['min_regime_size'] = np.min(regime_sizes)
            metrics['max_regime_size'] = np.max(regime_sizes)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {'error': str(e)}
    
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
            db_score = metrics.get('davies_bouldin_score', 0.0)
            regime_balance = metrics.get('regime_balance', 0.0)
            
            # Normalize scores
            normalized_silhouette = max(0, min(1, silhouette))
            normalized_ch = min(ch_score / 1000, 1.0)
            # Davies-Bouldin score should be minimized, so invert it
            normalized_db = max(0, min(1, 1.0 / (1.0 + db_score)))
            
            # Combined score with improved weighting
            score = (0.35 * normalized_silhouette + 
                    0.25 * normalized_ch + 
                    0.25 * normalized_db + 
                    0.15 * regime_balance)
            return score
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0


def create_unified_clustering_algorithm(config: Dict[str, Any]) -> UnifiedClusteringAlgorithm:
    """Create a unified clustering algorithm instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UnifiedClusteringAlgorithm instance
    """
    return UnifiedClusteringAlgorithm(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
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
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
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
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
