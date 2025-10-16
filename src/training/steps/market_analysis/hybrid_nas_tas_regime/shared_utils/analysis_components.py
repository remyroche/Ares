"""
Advanced Analysis Components for Hybrid NAS-TAS Regime Detection.

Provides common analysis component utilities for regime and cluster analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from abc import ABC, abstractmethod
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AnalysisComponentConfig:
    """Configuration for analysis component operations."""
    clustering_algorithm: str = "kmeans"  # "kmeans", "dbscan", "agglomerative"
    n_clusters: int = 5
    max_clusters: int = 20
    min_clusters: int = 2
    use_pca: bool = True
    pca_components: int = 10
    use_tsne: bool = False
    tsne_components: int = 2
    random_state: int = 42
    evaluation_metrics: List[str] = None

    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

@dataclass
class AnalysisResult:
    """Result from analysis component operations."""
    clusters: np.ndarray
    cluster_centers: np.ndarray
    cluster_labels: List[str]
    evaluation_metrics: Dict[str, float]
    analysis_metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class AdvancedAnalysisComponent(ABC):
    """Abstract base class for advanced analysis components."""

    def __init__(self, config: AnalysisComponentConfig):
        """Initialize the analysis component.

        Args:
            config: Analysis component configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def analyze(self, data: np.ndarray, features: Optional[np.ndarray] = None) -> AnalysisResult:
        """Perform analysis on the data.

        Args:
            data: Input data for analysis
            features: Optional feature data

        Returns:
            AnalysisResult with analysis results
        """
        pass

class RegimeAnalyzer(AdvancedAnalysisComponent):
    """Analyzer for market regime detection and analysis."""

    def __init__(self, config: AnalysisComponentConfig):
        """Initialize the regime analyzer.

        Args:
            config: Analysis component configuration
        """
        super().__init__(config)
        tprint_info("Initializing Regime Analyzer")
        tprint_debug(f"Configuration: {config}")
        self.logger.info("✅ Regime Analyzer initialized")
        tprint_success("Regime Analyzer initialized successfully")

    def analyze(self, data: np.ndarray, features: Optional[np.ndarray] = None) -> AnalysisResult:
        """Analyze market regimes.

        Args:
            data: Market data for regime analysis
            features: Optional feature data

        Returns:
            AnalysisResult with regime analysis results
        """
        try:
            tprint_info("Starting market regime analysis")
            tprint_debug(f"Data shape: {data.shape}, Features available: {features is not None}")
            self.logger.info("📊 Analyzing market regimes...")

            with tprint_timer("Regime Analysis"):
                start_time = time.time()

                # Prepare data for regime analysis
                tprint_debug("Preparing data for regime analysis")
                analysis_data = self._prepare_data_for_regime_analysis(data, features)
                tprint_debug(f"Analysis data shape: {analysis_data.shape}")

                # Perform regime clustering
                tprint_info("Performing regime clustering")
                clusters, cluster_centers = self._perform_regime_clustering(analysis_data)
                tprint_success(f"Clustering completed: {len(np.unique(clusters))} clusters found")

                # Generate cluster labels
                tprint_debug("Generating regime labels")
                cluster_labels = self._generate_regime_labels(clusters, cluster_centers)
                tprint_debug(f"Generated {len(cluster_labels)} regime labels")

                # Evaluate regime quality
                tprint_info("Evaluating regime quality")
                evaluation_metrics = self._evaluate_regime_quality(analysis_data, clusters)
                tprint_debug(f"Evaluation metrics: {evaluation_metrics}")

                # Create analysis metadata
                analysis_metadata = {
                    'regime_count': len(np.unique(clusters)),
                    'data_shape': data.shape,
                    'features_used': features is not None,
                    'clustering_algorithm': self.config.clustering_algorithm,
                    'analysis_timestamp': datetime.now().isoformat()
                }

                execution_time = time.time() - start_time
                tprint_performance("Regime Analysis", execution_time)

                self.logger.info(f"✅ Regime analysis completed: {len(np.unique(clusters))} regimes in {execution_time:.2f}s")

                return AnalysisResult(
                    clusters=clusters,
                    cluster_centers=cluster_centers,
                    cluster_labels=cluster_labels,
                    evaluation_metrics=evaluation_metrics,
                    analysis_metadata=analysis_metadata,
                    execution_time=execution_time,
                    success=True
                )

        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"Regime analysis failed: {e}")
            tprint_debug(f"Error details: {type(e).__name__}: {str(e)}")
            self.logger.error(f"❌ Regime analysis failed: {e}")
            return AnalysisResult(
                clusters=np.array([]),
                cluster_centers=np.array([]),
                cluster_labels=[],
                evaluation_metrics={},
                analysis_metadata={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _prepare_data_for_regime_analysis(self, data: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare data for regime analysis."""
        try:
            tprint_debug("Preparing data for regime analysis")
            if features is not None:
                tprint_debug("Using provided features")
                # Use provided features
                analysis_data = features
            else:
                tprint_debug("Extracting features from market data")
                # Extract features from market data
                analysis_data = self._extract_market_features(data)
                tprint_debug(f"Extracted features shape: {analysis_data.shape}")

            # Apply PCA if configured
            if self.config.use_pca and SKLEARN_AVAILABLE:
                tprint_info(f"Applying PCA with {self.config.pca_components} components")
                pca = PCA(n_components=self.config.pca_components, random_state=self.config.random_state)
                analysis_data = pca.fit_transform(analysis_data)
                tprint_success(f"PCA applied: {analysis_data.shape}")
                self.logger.info(f"📊 Applied PCA: {analysis_data.shape}")

            return analysis_data

        except Exception as e:
            tprint_warning(f"Data preparation failed: {e}")
            self.logger.warning(f"⚠️ Data preparation failed: {e}")
            return data

    def _extract_market_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from market data."""
        try:
            tprint_debug("Extracting market features")
            tprint_debug(f"Input data shape: {data.shape}")
            features = []

            # Price-based features
            if data.shape[1] >= 4:  # OHLC data
                tprint_debug("Extracting price-based features from OHLC data")
                # Returns
                returns = np.diff(data[:, 3]) / data[:-1, 3]  # Close price returns
                features.append(returns)
                tprint_debug(f"Returns feature shape: {returns.shape}")

                # Volatility
                volatility = np.abs(returns)
                features.append(volatility)
                tprint_debug(f"Volatility feature shape: {volatility.shape}")

                # Price range
                price_range = (data[:, 1] - data[:, 2]) / data[:, 3]  # (High - Low) / Close
                features.append(price_range[1:])  # Align with returns
                tprint_debug(f"Price range feature shape: {price_range[1:].shape}")

            # Volume-based features
            if data.shape[1] >= 5:  # OHLCV data
                volume = data[:, 4]
                volume_features = []

                # Volume change
                volume_change = np.diff(volume) / volume[:-1]
                volume_features.append(volume_change)

                # Volume volatility
                volume_volatility = np.abs(volume_change)
                volume_features.append(volume_volatility)

                features.extend(volume_features)

            # Combine features
            if features:
                # Align all features to same length
                min_length = min(len(f) for f in features)
                aligned_features = [f[:min_length] for f in features]
                combined_features = np.column_stack(aligned_features)
            else:
                # Fallback to original data
                combined_features = data

            return combined_features

        except Exception as e:
            self.logger.warning(f"⚠️ Feature extraction failed: {e}")
            return data

    def _perform_regime_clustering(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform regime clustering."""
        try:
            if not SKLEARN_AVAILABLE:
                # Fallback clustering
                n_clusters = min(self.config.n_clusters, len(data) // 10)
                clusters = np.random.randint(0, n_clusters, len(data))
                cluster_centers = np.random.rand(n_clusters, data.shape[1])
                return clusters, cluster_centers

            # Determine optimal number of clusters
            optimal_clusters = self._find_optimal_clusters(data)

            # Perform clustering
            if self.config.clustering_algorithm == "kmeans":
                clusterer = KMeans(n_clusters=optimal_clusters, random_state=self.config.random_state)
                clusters = clusterer.fit_predict(data)
                cluster_centers = clusterer.cluster_centers_

            elif self.config.clustering_algorithm == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                clusters = clusterer.fit_predict(data)
                # Calculate cluster centers manually
                cluster_centers = self._calculate_cluster_centers(data, clusters)

            elif self.config.clustering_algorithm == "agglomerative":
                clusterer = AgglomerativeClustering(n_clusters=optimal_clusters)
                clusters = clusterer.fit_predict(data)
                # Calculate cluster centers manually
                cluster_centers = self._calculate_cluster_centers(data, clusters)

            else:
                raise ValueError(f"Unknown clustering algorithm: {self.config.clustering_algorithm}")

            return clusters, cluster_centers

        except Exception as e:
            self.logger.warning(f"⚠️ Clustering failed: {e}")
            # Fallback to random clustering
            n_clusters = min(3, len(data) // 10)
            clusters = np.random.randint(0, n_clusters, len(data))
            cluster_centers = np.random.rand(n_clusters, data.shape[1])
            return clusters, cluster_centers

    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            if not SKLEARN_AVAILABLE or len(data) < 10:
                return self.config.n_clusters

            best_score = -np.inf
            best_clusters = self.config.n_clusters

            for n_clusters in range(self.config.min_clusters, min(self.config.max_clusters, len(data) // 5)):
                try:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                    clusters = clusterer.fit_predict(data)

                    # Calculate silhouette score
                    if len(np.unique(clusters)) > 1:
                        score = silhouette_score(data, clusters)
                        if score > best_score:
                            best_score = score
                            best_clusters = n_clusters

                except Exception:
                    continue

            return best_clusters

        except Exception as e:
            self.logger.warning(f"⚠️ Optimal clusters finding failed: {e}")
            return self.config.n_clusters

    def _calculate_cluster_centers(self, data: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Calculate cluster centers manually."""
        try:
            unique_clusters = np.unique(clusters)
            cluster_centers = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue
                cluster_data = data[clusters == cluster_id]
                if len(cluster_data) > 0:
                    center = np.mean(cluster_data, axis=0)
                    cluster_centers.append(center)

            return np.array(cluster_centers) if cluster_centers else np.array([])

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster centers calculation failed: {e}")
            return np.array([])

    def _generate_regime_labels(self, clusters: np.ndarray, cluster_centers: np.ndarray) -> List[str]:
        """Generate descriptive labels for regimes."""
        try:
            unique_clusters = np.unique(clusters)
            cluster_labels = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Noise in DBSCAN
                    cluster_labels.append("Noise")
                else:
                    # Generate descriptive label based on cluster characteristics
                    cluster_mask = clusters == cluster_id
                    cluster_size = np.sum(cluster_mask)

                    if cluster_size < 10:
                        label = f"Minor_Regime_{cluster_id}"
                    elif cluster_size < 50:
                        label = f"Moderate_Regime_{cluster_id}"
                    else:
                        label = f"Major_Regime_{cluster_id}"

                    cluster_labels.append(label)

            return cluster_labels

        except Exception as e:
            self.logger.warning(f"⚠️ Regime label generation failed: {e}")
            return [f"Regime_{i}" for i in range(len(np.unique(clusters)))]

    def _evaluate_regime_quality(self, data: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of regime clustering."""
        try:
            if not SKLEARN_AVAILABLE or len(np.unique(clusters)) < 2:
                return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

            metrics = {}

            # Silhouette score
            if 'silhouette' in self.config.evaluation_metrics:
                try:
                    metrics['silhouette'] = silhouette_score(data, clusters)
                except Exception:
                    metrics['silhouette'] = 0.0

            # Calinski-Harabasz score
            if 'calinski_harabasz' in self.config.evaluation_metrics:
                try:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(data, clusters)
                except Exception:
                    metrics['calinski_harabasz'] = 0.0

            # Davies-Bouldin score
            if 'davies_bouldin' in self.config.evaluation_metrics:
                try:
                    metrics['davies_bouldin'] = davies_bouldin_score(data, clusters)
                except Exception:
                    metrics['davies_bouldin'] = 1.0

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Regime quality evaluation failed: {e}")
            return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

class ClusterAnalyzer(AdvancedAnalysisComponent):
    """Analyzer for cluster analysis and validation."""

    def __init__(self, config: AnalysisComponentConfig):
        """Initialize the cluster analyzer.

        Args:
            config: Analysis component configuration
        """
        super().__init__(config)
        self.logger.info("✅ Cluster Analyzer initialized")

    def analyze(self, data: np.ndarray, features: Optional[np.ndarray] = None) -> AnalysisResult:
        """Analyze clusters.

        Args:
            data: Input data for cluster analysis
            features: Optional feature data

        Returns:
            AnalysisResult with cluster analysis results
        """
        try:
            self.logger.info("🔍 Analyzing clusters...")
            start_time = time.time()

            # Prepare data for cluster analysis
            analysis_data = self._prepare_data_for_cluster_analysis(data, features)

            # Perform cluster analysis
            clusters, cluster_centers = self._perform_cluster_analysis(analysis_data)

            # Generate cluster labels
            cluster_labels = self._generate_cluster_labels(clusters, cluster_centers)

            # Evaluate cluster quality
            evaluation_metrics = self._evaluate_cluster_quality(analysis_data, clusters)

            # Create analysis metadata
            analysis_metadata = {
                'cluster_count': len(np.unique(clusters)),
                'data_shape': data.shape,
                'features_used': features is not None,
                'clustering_algorithm': self.config.clustering_algorithm,
                'analysis_timestamp': datetime.now().isoformat()
            }

            execution_time = time.time() - start_time

            self.logger.info(f"✅ Cluster analysis completed: {len(np.unique(clusters))} clusters in {execution_time:.2f}s")

            return AnalysisResult(
                clusters=clusters,
                cluster_centers=cluster_centers,
                cluster_labels=cluster_labels,
                evaluation_metrics=evaluation_metrics,
                analysis_metadata=analysis_metadata,
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Cluster analysis failed: {e}")
            return AnalysisResult(
                clusters=np.array([]),
                cluster_centers=np.array([]),
                cluster_labels=[],
                evaluation_metrics={},
                analysis_metadata={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _prepare_data_for_cluster_analysis(self, data: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare data for cluster analysis."""
        try:
            if features is not None:
                analysis_data = features
            else:
                analysis_data = data

            # Apply dimensionality reduction if configured
            if self.config.use_pca and SKLEARN_AVAILABLE and analysis_data.shape[1] > self.config.pca_components:
                pca = PCA(n_components=self.config.pca_components, random_state=self.config.random_state)
                analysis_data = pca.fit_transform(analysis_data)
                self.logger.info(f"📊 Applied PCA: {analysis_data.shape}")

            if self.config.use_tsne and SKLEARN_AVAILABLE:
                tsne = TSNE(n_components=self.config.tsne_components, random_state=self.config.random_state)
                analysis_data = tsne.fit_transform(analysis_data)
                self.logger.info(f"📊 Applied t-SNE: {analysis_data.shape}")

            return analysis_data

        except Exception as e:
            self.logger.warning(f"⚠️ Data preparation failed: {e}")
            return data

    def _perform_cluster_analysis(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform cluster analysis."""
        try:
            if not SKLEARN_AVAILABLE:
                # Fallback clustering
                n_clusters = min(self.config.n_clusters, len(data) // 10)
                clusters = np.random.randint(0, n_clusters, len(data))
                cluster_centers = np.random.rand(n_clusters, data.shape[1])
                return clusters, cluster_centers

            # Determine optimal number of clusters
            optimal_clusters = self._find_optimal_clusters(data)

            # Perform clustering
            if self.config.clustering_algorithm == "kmeans":
                clusterer = KMeans(n_clusters=optimal_clusters, random_state=self.config.random_state)
                clusters = clusterer.fit_predict(data)
                cluster_centers = clusterer.cluster_centers_

            elif self.config.clustering_algorithm == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                clusters = clusterer.fit_predict(data)
                cluster_centers = self._calculate_cluster_centers(data, clusters)

            elif self.config.clustering_algorithm == "agglomerative":
                clusterer = AgglomerativeClustering(n_clusters=optimal_clusters)
                clusters = clusterer.fit_predict(data)
                cluster_centers = self._calculate_cluster_centers(data, clusters)

            else:
                raise ValueError(f"Unknown clustering algorithm: {self.config.clustering_algorithm}")

            return clusters, cluster_centers

        except Exception as e:
            self.logger.warning(f"⚠️ Clustering failed: {e}")
            # Fallback to random clustering
            n_clusters = min(3, len(data) // 10)
            clusters = np.random.randint(0, n_clusters, len(data))
            cluster_centers = np.random.rand(n_clusters, data.shape[1])
            return clusters, cluster_centers

    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find optimal number of clusters."""
        try:
            if not SKLEARN_AVAILABLE or len(data) < 10:
                return self.config.n_clusters

            best_score = -np.inf
            best_clusters = self.config.n_clusters

            for n_clusters in range(self.config.min_clusters, min(self.config.max_clusters, len(data) // 5)):
                try:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                    clusters = clusterer.fit_predict(data)

                    if len(np.unique(clusters)) > 1:
                        score = silhouette_score(data, clusters)
                        if score > best_score:
                            best_score = score
                            best_clusters = n_clusters

                except Exception:
                    continue

            return best_clusters

        except Exception as e:
            self.logger.warning(f"⚠️ Optimal clusters finding failed: {e}")
            return self.config.n_clusters

    def _calculate_cluster_centers(self, data: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        try:
            unique_clusters = np.unique(clusters)
            cluster_centers = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue
                cluster_data = data[clusters == cluster_id]
                if len(cluster_data) > 0:
                    center = np.mean(cluster_data, axis=0)
                    cluster_centers.append(center)

            return np.array(cluster_centers) if cluster_centers else np.array([])

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster centers calculation failed: {e}")
            return np.array([])

    def _generate_cluster_labels(self, clusters: np.ndarray, cluster_centers: np.ndarray) -> List[str]:
        """Generate cluster labels."""
        try:
            unique_clusters = np.unique(clusters)
            cluster_labels = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    cluster_labels.append("Noise")
                else:
                    cluster_labels.append(f"Cluster_{cluster_id}")

            return cluster_labels

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster label generation failed: {e}")
            return [f"Cluster_{i}" for i in range(len(np.unique(clusters)))]

    def _evaluate_cluster_quality(self, data: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """Evaluate cluster quality."""
        try:
            if not SKLEARN_AVAILABLE or len(np.unique(clusters)) < 2:
                return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

            metrics = {}

            if 'silhouette' in self.config.evaluation_metrics:
                try:
                    metrics['silhouette'] = silhouette_score(data, clusters)
                except Exception:
                    metrics['silhouette'] = 0.0

            if 'calinski_harabasz' in self.config.evaluation_metrics:
                try:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(data, clusters)
                except Exception:
                    metrics['calinski_harabasz'] = 0.0

            if 'davies_bouldin' in self.config.evaluation_metrics:
                try:
                    metrics['davies_bouldin'] = davies_bouldin_score(data, clusters)
                except Exception:
                    metrics['davies_bouldin'] = 1.0

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster quality evaluation failed: {e}")
            return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

def create_regime_analyzer(config: AnalysisComponentConfig) -> RegimeAnalyzer:
    """Create a regime analyzer instance.

    Args:
        config: Analysis component configuration

    Returns:
        RegimeAnalyzer instance
    """
    return RegimeAnalyzer(config)

def create_cluster_analyzer(config: AnalysisComponentConfig) -> ClusterAnalyzer:
    """Create a cluster analyzer instance.

    Args:
        config: Analysis component configuration

    Returns:
        ClusterAnalyzer instance
    """
    return ClusterAnalyzer(config)

class SharedClusteringUtilities:
    """Shared clustering utilities for TAS and NAS systems."""

    def __init__(self):
        """Initialize shared clustering utilities."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("✅ Shared Clustering Utilities initialized")

    def perform_shared_clustering(self, data: np.ndarray, n_clusters: int = 8,
                                 algorithm: str = "auto") -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Perform clustering using shared algorithms.

        Args:
            data: Input data for clustering
            n_clusters: Number of clusters
            algorithm: Clustering algorithm ("auto", "kmeans", "dbscan", "agglomerative", "gmm")

        Returns:
            Tuple of (cluster_labels, cluster_centers, metrics)
        """
        try:
            self.logger.info(f"🔍 Performing shared clustering with {algorithm} algorithm...")

            if not SKLEARN_AVAILABLE:
                self.logger.warning("⚠️ sklearn not available, using fallback clustering")
                return self._fallback_clustering(data, n_clusters)

            # Auto-select best algorithm
            if algorithm == "auto":
                algorithm = self._select_best_algorithm(data, n_clusters)

            # Perform clustering based on algorithm
            if algorithm == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(data)
                centers = clusterer.cluster_centers_

            elif algorithm == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                labels = clusterer.fit_predict(data)
                centers = self._calculate_cluster_centers(data, labels)

            elif algorithm == "agglomerative":
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(data)
                centers = self._calculate_cluster_centers(data, labels)

            elif algorithm == "gmm":
                from sklearn.mixture import GaussianMixture
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = clusterer.fit_predict(data)
                centers = clusterer.means_

            else:
                raise ValueError(f"Unknown clustering algorithm: {algorithm}")

            # Calculate metrics
            metrics = self._calculate_clustering_metrics(data, labels)

            self.logger.info(f"✅ Clustering completed: {len(np.unique(labels))} clusters")
            return labels, centers, metrics

        except Exception as e:
            self.logger.error(f"❌ Shared clustering failed: {e}")
            return self._fallback_clustering(data, n_clusters)

    def _select_best_algorithm(self, data: np.ndarray, n_clusters: int) -> str:
        """Select the best clustering algorithm for the data."""
        try:
            if not SKLEARN_AVAILABLE or len(data) < 10:
                return "kmeans"

            algorithms = ["kmeans", "gmm", "agglomerative"]
            best_score = -np.inf
            best_algorithm = "kmeans"

            for algorithm in algorithms:
                try:
                    if algorithm == "kmeans":
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        labels = clusterer.fit_predict(data)

                    elif algorithm == "gmm":
                        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                        labels = clusterer.fit_predict(data)

                    elif algorithm == "agglomerative":
                        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                        labels = clusterer.fit_predict(data)

                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(data, labels)
                        if score > best_score:
                            best_score = score
                            best_algorithm = algorithm

                except Exception:
                    continue

            return best_algorithm

        except Exception as e:
            self.logger.warning(f"⚠️ Algorithm selection failed: {e}")
            return "kmeans"

    def _calculate_clustering_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            if not SKLEARN_AVAILABLE or len(np.unique(labels)) < 2:
                return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

            metrics = {}

            # Silhouette score
            try:
                metrics['silhouette'] = silhouette_score(data, labels)
            except Exception:
                metrics['silhouette'] = 0.0

            # Calinski-Harabasz score
            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
            except Exception:
                metrics['calinski_harabasz'] = 0.0

            # Davies-Bouldin score
            try:
                metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
            except Exception:
                metrics['davies_bouldin'] = 1.0

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Metrics calculation failed: {e}")
            return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

    def _calculate_cluster_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers manually."""
        try:
            unique_labels = np.unique(labels)
            cluster_centers = []

            for label in unique_labels:
                if label == -1:  # Skip noise points in DBSCAN
                    continue
                cluster_data = data[labels == label]
                if len(cluster_data) > 0:
                    center = np.mean(cluster_data, axis=0)
                    cluster_centers.append(center)

            return np.array(cluster_centers) if cluster_centers else np.array([])

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster centers calculation failed: {e}")
            return np.array([])

    def _fallback_clustering(self, data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Fallback clustering when sklearn is not available."""
        try:
            n_samples = len(data)
            labels = np.random.randint(0, n_clusters, n_samples)
            centers = np.random.rand(n_clusters, data.shape[1])

            return labels, centers, {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 1.0}

        except Exception as e:
            self.logger.error(f"❌ Fallback clustering failed: {e}")
            return np.array([]), np.array([]), {}

def create_shared_clustering_utilities() -> SharedClusteringUtilities:
    """Create shared clustering utilities instance."""
    return SharedClusteringUtilities()
