"""
NAS-driven clusterer for short-term trading regime detection.

This module provides the main NAS clustering functionality optimized for
short-term trading with micro-regime detection capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import talib

from .nas_config import NASClusteringConfig, NASArchitectureType
from .nas_feature_extractor import NASFeatureExtractor, NASFeatureResult
from .micro_regime_detector import MicroRegimeDetector, MicroRegimeResult

logger = logging.getLogger(__name__)


@dataclass
class NASClusteringResult:
    """Result of NAS clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    statistics: Dict[str, Any]
    quality_metrics: Dict[str, float]
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    # NAS-specific fields
    nas_architectures: Dict[str, Any] = None
    micro_regimes: Optional[MicroRegimeResult] = None
    regime_transitions: np.ndarray = None
    economic_significance_scores: np.ndarray = None
    trading_viability_scores: np.ndarray = None


class NASClusterer:
    """NAS-driven clusterer for short-term trading regime detection."""
    
    def __init__(self, config: NASClusteringConfig):
        """Initialize NAS clusterer.
        
        Args:
            config: NAS clustering configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.feature_extractor = NASFeatureExtractor(config.get_feature_config())
        self.micro_regime_detector = MicroRegimeDetector(config.get_micro_regime_config())
        
        # NAS architecture settings
        self.nas_architecture_type = config.nas_architecture_type
        self.n_regimes = config.n_regimes
        self.min_regime_duration = config.min_regime_duration
        self.max_regime_duration = config.max_regime_duration
        
        # Economic significance settings
        self.economic_significance_threshold = config.economic_significance_threshold
        self.trading_viability_threshold = config.trading_viability_threshold
        self.regime_transition_cost = config.regime_transition_cost
        
        # Initialize NAS architectures
        self.nas_architectures = self._initialize_nas_architectures()
        
        self.logger.info(f"✅ NAS Clusterer initialized for {config.timeframe} timeframe with {config.n_regimes} regimes")
    
    def cluster(self, data: Union[pd.DataFrame, np.ndarray], 
                timestamps: Optional[np.ndarray] = None,
                optimize_parameters: bool = True,
                generate_report: bool = True) -> NASClusteringResult:
        """Perform NAS-driven clustering on market data.
        
        Args:
            data: Market data (DataFrame or numpy array)
            timestamps: Optional timestamps array
            optimize_parameters: Whether to optimize NAS parameters
            generate_report: Whether to generate clustering report
            
        Returns:
            NASClusteringResult with clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting NAS-driven clustering")
            
            # Prepare data
            if isinstance(data, pd.DataFrame):
                data_array = data.values
                if timestamps is None and 'timestamp' in data.columns:
                    timestamps = data['timestamp'].values
            else:
                data_array = data
                if timestamps is None:
                    timestamps = np.arange(len(data))
            
            # Extract NAS features
            feature_result = self.feature_extractor.extract_features(data_array, timestamps)
            
            # Detect micro-regimes
            micro_regime_result = self.micro_regime_detector.detect_micro_regimes(
                data_array, timestamps, feature_result.features
            )
            
            # Perform NAS clustering
            clustering_result = self._perform_nas_clustering(
                feature_result, micro_regime_result, optimize_parameters
            )
            
            # Calculate economic significance and trading viability
            economic_scores = self._calculate_economic_significance_scores(
                data_array, clustering_result['labels']
            )
            trading_scores = self._calculate_trading_viability_scores(
                data_array, clustering_result['labels']
            )
            
            # Calculate regime transitions
            regime_transitions = self._calculate_regime_transitions(
                clustering_result['labels']
            )
            
            # Create NAS clustering result
            nas_result = self._create_nas_result(
                clustering_result, micro_regime_result, economic_scores,
                trading_scores, regime_transitions, feature_result,
                time.time() - start_time
            )
            
            # Generate report if requested
            if generate_report:
                self._generate_nas_report(nas_result, feature_result)
            
            self.logger.info(f"✅ NAS clustering completed in {nas_result.execution_time:.2f}s")
            return nas_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS clustering failed: {e}")
            return NASClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e), 'method': 'nas_clustering'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _initialize_nas_architectures(self) -> Dict[str, Any]:
        """Initialize NAS architectures for different regime types."""
        try:
            architectures = {}
            
            # Volatility-focused architecture
            architectures['volatility'] = {
                'type': 'volatility_focused',
                'layers': [64, 32, 16],
                'activation': 'relu',
                'dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }
            
            # Trend-focused architecture
            architectures['trend'] = {
                'type': 'trend_focused',
                'layers': [128, 64, 32],
                'activation': 'tanh',
                'dropout': 0.3,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }
            
            # Volume-focused architecture
            architectures['volume'] = {
                'type': 'volume_focused',
                'layers': [32, 16, 8],
                'activation': 'relu',
                'dropout': 0.1,
                'optimizer': 'adam',
                'learning_rate': 0.01
            }
            
            # Momentum-focused architecture
            architectures['momentum'] = {
                'type': 'momentum_focused',
                'layers': [96, 48, 24],
                'activation': 'swish',
                'dropout': 0.25,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }
            
            # Hybrid architecture
            architectures['hybrid'] = {
                'type': 'hybrid',
                'layers': [256, 128, 64, 32],
                'activation': 'relu',
                'dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }
            
            return architectures
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS architecture initialization failed: {e}")
            return {}
    
    def _perform_nas_clustering(self, feature_result: NASFeatureResult,
                              micro_regime_result: MicroRegimeResult,
                              optimize_parameters: bool) -> Dict[str, Any]:
        """Perform NAS-driven clustering."""
        try:
            features = feature_result.features
            if features.size == 0:
                raise ValueError("No features available for clustering")
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Select clustering method based on NAS architecture
            clustering_method = self._select_clustering_method(features_scaled)
            
            # Perform clustering
            if clustering_method == 'kmeans':
                clusterer = KMeans(
                    n_clusters=self.n_regimes,
                    random_state=42,
                    n_init=10
                )
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = clusterer.cluster_centers_
                
            elif clustering_method == 'agglomerative':
                clusterer = AgglomerativeClustering(
                    n_clusters=self.n_regimes,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = self._calculate_cluster_centers(features_scaled, labels)
                
            elif clustering_method == 'dbscan':
                clusterer = DBSCAN(
                    eps=0.5,
                    min_samples=self.min_regime_duration
                )
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = self._calculate_cluster_centers(features_scaled, labels)
                
            else:
                # Default to K-means
                clusterer = KMeans(
                    n_clusters=self.n_regimes,
                    random_state=42,
                    n_init=10
                )
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = clusterer.cluster_centers_
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(features_scaled, labels)
            
            # Calculate statistics
            statistics = self._calculate_clustering_statistics(labels, cluster_centers)
            
            # Validate clustering
            validation = self._validate_clustering(labels, quality_metrics)
            
            return {
                'labels': labels,
                'cluster_centers': cluster_centers,
                'quality_metrics': quality_metrics,
                'statistics': statistics,
                'validation': validation,
                'clustering_method': clustering_method
            }
            
        except Exception as e:
            self.logger.error(f"❌ NAS clustering failed: {e}")
            raise
    
    def _select_clustering_method(self, features: np.ndarray) -> str:
        """Select optimal clustering method based on NAS architecture."""
        try:
            # Use NAS architecture type to select method
            if self.nas_architecture_type == NASArchitectureType.VOLATILITY_FOCUSED:
                return 'kmeans'  # Good for volatility patterns
            elif self.nas_architecture_type == NASArchitectureType.TREND_FOCUSED:
                return 'agglomerative'  # Good for trend patterns
            elif self.nas_architecture_type == NASArchitectureType.VOLUME_FOCUSED:
                return 'dbscan'  # Good for volume patterns
            elif self.nas_architecture_type == NASArchitectureType.MOMENTUM_FOCUSED:
                return 'kmeans'  # Good for momentum patterns
            else:  # HYBRID
                return 'kmeans'  # Default to K-means for hybrid
                
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering method selection failed: {e}")
            return 'kmeans'
    
    def _calculate_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers from labels."""
        try:
            unique_labels = np.unique(labels)
            cluster_centers = []
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                cluster_mask = labels == label
                cluster_center = np.mean(features[cluster_mask], axis=0)
                cluster_centers.append(cluster_center)
            
            return np.array(cluster_centers) if cluster_centers else np.array([])
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster center calculation failed: {e}")
            return np.array([])
    
    def _calculate_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            if len(np.unique(labels)) < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
            
            # Silhouette score
            silhouette = silhouette_score(features, labels)
            
            # Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(features, labels)
            
            # Custom NAS metrics
            nas_score = self._calculate_nas_score(features, labels)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'nas_score': nas_score
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality metrics calculation failed: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'nas_score': 0.0}
    
    def _calculate_nas_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate custom NAS score for regime quality."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(labels)
            
            # Calculate regime separation
            regime_separation = self._calculate_regime_separation(features, labels)
            
            # Calculate regime consistency
            regime_consistency = self._calculate_regime_consistency(features, labels)
            
            # Combined NAS score
            nas_score = (regime_stability + regime_separation + regime_consistency) / 3.0
            
            return nas_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS score calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_stability(self, labels: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            # Calculate regime persistence
            regime_changes = np.sum(np.diff(labels) != 0)
            total_periods = len(labels) - 1
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return stability
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime separation score."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate inter-cluster distance
            inter_cluster_distances = []
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    if i < j:
                        cluster1_mask = labels == label1
                        cluster2_mask = labels == label2
                        
                        if np.any(cluster1_mask) and np.any(cluster2_mask):
                            center1 = np.mean(features[cluster1_mask], axis=0)
                            center2 = np.mean(features[cluster2_mask], axis=0)
                            distance = np.linalg.norm(center1 - center2)
                            inter_cluster_distances.append(distance)
            
            # Calculate intra-cluster distance
            intra_cluster_distances = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_features = features[cluster_mask]
                    center = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    intra_cluster_distances.extend(distances)
            
            # Calculate separation ratio
            if inter_cluster_distances and intra_cluster_distances:
                avg_inter = np.mean(inter_cluster_distances)
                avg_intra = np.mean(intra_cluster_distances)
                separation = avg_inter / (avg_intra + 1e-8)
                return min(separation, 1.0)  # Cap at 1.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime separation calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_consistency(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime consistency score."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate consistency within each regime
            consistency_scores = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_features = features[cluster_mask]
                    # Calculate feature variance within cluster
                    feature_variance = np.var(cluster_features, axis=0)
                    # Lower variance = higher consistency
                    consistency = 1.0 / (1.0 + np.mean(feature_variance))
                    consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime consistency calculation failed: {e}")
            return 0.0
    
    def _calculate_clustering_statistics(self, labels: np.ndarray, 
                                       cluster_centers: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering statistics."""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Regime distribution
            regime_distribution = {}
            for label in unique_labels:
                count = np.sum(labels == label)
                regime_distribution[f'regime_{label}'] = count
            
            # Regime percentages
            regime_percentages = {}
            total_samples = len(labels)
            for label in unique_labels:
                count = np.sum(labels == label)
                percentage = (count / total_samples) * 100
                regime_percentages[f'regime_{label}'] = percentage
            
            return {
                'n_clusters': n_clusters,
                'regime_distribution': regime_distribution,
                'regime_percentages': regime_percentages,
                'total_samples': total_samples,
                'cluster_centers_shape': cluster_centers.shape if cluster_centers.size > 0 else (0, 0)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering statistics calculation failed: {e}")
            return {}
    
    def _validate_clustering(self, labels: np.ndarray, 
                           quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate clustering results."""
        try:
            # Check minimum quality thresholds
            silhouette_threshold = 0.3
            nas_score_threshold = 0.4
            
            silhouette_valid = quality_metrics.get('silhouette_score', 0.0) >= silhouette_threshold
            nas_score_valid = quality_metrics.get('nas_score', 0.0) >= nas_score_threshold
            
            # Check regime count
            unique_labels = np.unique(labels)
            regime_count_valid = 5 <= len(unique_labels) <= 20
            
            # Overall validation
            is_valid = silhouette_valid and nas_score_valid and regime_count_valid
            
            return {
                'valid': is_valid,
                'silhouette_valid': silhouette_valid,
                'nas_score_valid': nas_score_valid,
                'regime_count_valid': regime_count_valid,
                'n_regimes': len(unique_labels),
                'silhouette_score': quality_metrics.get('silhouette_score', 0.0),
                'nas_score': quality_metrics.get('nas_score', 0.0)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _calculate_economic_significance_scores(self, data: np.ndarray, 
                                              labels: np.ndarray) -> np.ndarray:
        """Calculate economic significance scores for each regime."""
        try:
            if data.shape[1] < 4:
                return np.zeros(len(labels))
            
            close_price = data[:, 3]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            economic_scores = np.zeros(len(labels))
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue
                
                regime_close = close_price[regime_mask]
                regime_volume = volume[regime_mask]
                
                # Calculate economic significance based on price movement and volume
                price_change = abs((regime_close[-1] - regime_close[0]) / regime_close[0])
                volume_ratio = np.mean(regime_volume) / np.mean(volume)
                volatility = np.std(regime_close) / np.mean(regime_close)
                
                # Economic significance score
                economic_score = (price_change * volume_ratio * volatility) / 3.0
                economic_scores[regime_mask] = min(economic_score, 1.0)
            
            return economic_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic significance calculation failed: {e}")
            return np.zeros(len(labels))
    
    def _calculate_trading_viability_scores(self, data: np.ndarray, 
                                          labels: np.ndarray) -> np.ndarray:
        """Calculate trading viability scores for each regime."""
        try:
            if data.shape[1] < 4:
                return np.zeros(len(labels))
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            trading_scores = np.zeros(len(labels))
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue
                
                regime_close = close_price[regime_mask]
                regime_high = high_price[regime_mask]
                regime_low = low_price[regime_mask]
                regime_volume = volume[regime_mask]
                
                # Calculate trading viability based on multiple factors
                price_range = (np.max(regime_high) - np.min(regime_low)) / np.mean(regime_close)
                volume_consistency = 1.0 / (1.0 + np.std(regime_volume) / np.mean(regime_volume))
                trend_consistency = self._calculate_trend_consistency(regime_close)
                
                # Trading viability score
                trading_score = (price_range * volume_consistency * trend_consistency) / 3.0
                trading_scores[regime_mask] = min(trading_score, 1.0)
            
            return trading_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading viability calculation failed: {e}")
            return np.zeros(len(labels))
    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate trend consistency for a price series."""
        try:
            if len(prices) < 3:
                return 0.5
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Calculate trend consistency
            positive_changes = np.sum(price_changes > 0)
            negative_changes = np.sum(price_changes < 0)
            total_changes = len(price_changes)
            
            if total_changes == 0:
                return 0.5
            
            # Consistency is higher when there's a clear trend direction
            consistency = max(positive_changes, negative_changes) / total_changes
            return consistency
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_regime_transitions(self, labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition probabilities."""
        try:
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)
            
            if n_regimes < 2:
                return np.array([])
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                
                if current_regime in unique_labels and next_regime in unique_labels:
                    current_idx = np.where(unique_labels == current_regime)[0][0]
                    next_idx = np.where(unique_labels == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition calculation failed: {e}")
            return np.array([])
    
    def _create_nas_result(self, clustering_result: Dict[str, Any],
                          micro_regime_result: MicroRegimeResult,
                          economic_scores: np.ndarray,
                          trading_scores: np.ndarray,
                          regime_transitions: np.ndarray,
                          feature_result: NASFeatureResult,
                          execution_time: float) -> NASClusteringResult:
        """Create NAS clustering result."""
        try:
            from datetime import datetime
            
            # Create base result
            result = NASClusteringResult(
                labels=clustering_result['labels'],
                cluster_centers=clustering_result['cluster_centers'],
                statistics=clustering_result['statistics'],
                quality_metrics=clustering_result['quality_metrics'],
                validation=clustering_result['validation'],
                metadata={
                    'method': 'nas_clustering',
                    'timeframe': self.config.timeframe,
                    'n_regimes': self.n_regimes,
                    'nas_architecture_type': self.nas_architecture_type.value,
                    'micro_regime_detection': self.config.enable_micro_regime_detection,
                    'feature_count': len(feature_result.feature_names),
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                },
                success=clustering_result['validation']['valid'],
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                
                # NAS-specific fields
                nas_architectures=self.nas_architectures,
                micro_regimes=micro_regime_result,
                regime_transitions=regime_transitions,
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ NAS result creation failed: {e}")
            return NASClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _generate_nas_report(self, result: NASClusteringResult, 
                           feature_result: NASFeatureResult) -> None:
        """Generate NAS clustering report."""
        try:
            self.logger.info("📊 NAS Clustering Report:")
            self.logger.info(f"   - Method: {result.metadata.get('method', 'nas_clustering')}")
            self.logger.info(f"   - Timeframe: {result.metadata.get('timeframe', '15m')}")
            self.logger.info(f"   - Regimes: {result.metadata.get('n_regimes', 0)}")
            self.logger.info(f"   - Architecture: {result.metadata.get('nas_architecture_type', 'hybrid')}")
            self.logger.info(f"   - Silhouette Score: {result.quality_metrics.get('silhouette_score', 0.0):.4f}")
            self.logger.info(f"   - NAS Score: {result.quality_metrics.get('nas_score', 0.0):.4f}")
            self.logger.info(f"   - Economic Significance: {np.mean(result.economic_significance_scores):.4f}")
            self.logger.info(f"   - Trading Viability: {np.mean(result.trading_viability_scores):.4f}")
            self.logger.info(f"   - Micro-regimes: {len(result.micro_regimes.micro_regime_types) if result.micro_regimes else 0}")
            self.logger.info(f"   - Execution Time: {result.execution_time:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS report generation failed: {e}")