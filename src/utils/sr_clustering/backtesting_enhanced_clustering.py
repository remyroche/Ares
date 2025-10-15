"""
Backtesting Enhanced Clustering

This module provides clustering algorithms enhanced with backtesting validation
for Support/Resistance level optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ..logger import system_logger

# Import SR backtesting engine
from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel, BacktestResult

# Import optimization utilities
try:
    from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from ..matrix_operations import get_unified_matrix_operations, M1EnhancedMatrixOperations
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False
    get_m1_memory_optimizer = None
    get_unified_matrix_operations = None

logger = logging.getLogger(__name__)


@dataclass
class BacktestingEnhancedConfig:
    """Configuration for backtesting-enhanced clustering."""

    # Clustering parameters
    clustering_method: str = 'dbscan'  # 'dbscan', 'agglomerative', 'hybrid'
    eps: float = 0.02  # DBSCAN epsilon parameter
    min_samples: int = 5  # DBSCAN minimum samples
    n_clusters: int = 10  # Agglomerative clustering number of clusters

    # Backtesting integration
    enable_backtesting_validation: bool = True
    backtest_config: Optional[BacktestConfig] = None
    min_backtest_score: float = 0.1  # Minimum Sharpe ratio for cluster validation

    # Feature engineering
    use_price_features: bool = True
    use_volume_features: bool = True
    use_time_features: bool = True
    feature_normalization: str = 'standard'  # 'standard', 'minmax', 'robust'

    # Hardware optimization
    enable_parallel_clustering: bool = True
    enable_memory_optimization: bool = True
    max_memory_gb: float = 8.0

    # Validation parameters
    validation_split: float = 0.2  # 20% for validation
    cross_validation_folds: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.clustering_method not in ['dbscan', 'agglomerative', 'hybrid']:
            raise ValueError(f"Invalid clustering_method: {self.clustering_method}")

        if self.eps <= 0:
            raise ValueError("eps must be positive")

        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")


@dataclass
class ClusterResult:
    """Result from clustering operation."""

    cluster_id: int
    level_indices: List[int]
    centroid_price: float
    cluster_size: int
    silhouette_score: float
    backtest_score: Optional[float] = None
    confidence: float = 1.0


class BacktestingEnhancedClustering:
    """
    Clustering algorithm enhanced with backtesting validation.

    This class performs clustering on price data and validates clusters
    using backtesting to ensure economic significance.
    """

    def __init__(self, config: BacktestingEnhancedConfig):
        """Initialize the backtesting-enhanced clustering."""
        self.config = config
        self.logger = system_logger.getChild('BacktestingEnhancedClustering')

        # Setup backtesting engine if enabled
        self.backtesting_engine = None
        if self.config.enable_backtesting_validation:
            if self.config.backtest_config is None:
                self.config.backtest_config = BacktestConfig()
            self.backtesting_engine = SRBacktestingEngine(self.config.backtest_config)

        # Hardware optimization setup
        self._setup_hardware_optimizations()

        self.logger.info("✅ Backtesting Enhanced Clustering initialized")

    def _setup_hardware_optimizations(self):
        """Setup hardware optimizations."""
        if not M1_OPTIMIZATIONS_AVAILABLE:
            self.logger.warning("M1 optimizations not available")
            return

        try:
            if self.config.enable_memory_optimization:
                self.memory_optimizer = get_m1_memory_optimizer() if get_m1_memory_optimizer else None
                if self.memory_optimizer:
                    self.memory_optimizer.set_memory_limit(self.config.max_memory_gb * 1024**3)
        except Exception as e:
            self.logger.warning(f"Failed to setup hardware optimizations: {e}")

    async def cluster_and_validate(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        existing_levels: Optional[List[SRLevel]] = None
    ) -> List[ClusterResult]:
        """
        Perform clustering with backtesting validation.

        Args:
            price_data: OHLC price data
            volume_data: Volume data (optional)
            existing_levels: Existing SR levels for incremental clustering

        Returns:
            List of validated cluster results
        """
        self.logger.info("🔄 Starting clustering with backtesting validation")

        try:
            # Extract features for clustering
            features = self._extract_clustering_features(price_data, volume_data)

            if features is None or features.empty:
                self.logger.error("Failed to extract clustering features")
                return []

            # Perform clustering
            cluster_labels = self._perform_clustering(features)

            if cluster_labels is None:
                self.logger.error("Clustering failed")
                return []

            # Create cluster results
            cluster_results = self._create_cluster_results(
                features, cluster_labels, price_data.index
            )

            # Validate clusters with backtesting if enabled
            if self.config.enable_backtesting_validation and self.backtesting_engine:
                validated_results = await self._validate_clusters_with_backtesting(
                    cluster_results, price_data, volume_data
                )
            else:
                validated_results = cluster_results

            # Filter by minimum backtest score if validation enabled
            if self.config.enable_backtesting_validation:
                validated_results = [
                    result for result in validated_results
                    if result.backtest_score is None or result.backtest_score >= self.config.min_backtest_score
                ]

            self.logger.info(f"✅ Clustering completed: {len(validated_results)} validated clusters")
            return validated_results

        except Exception as e:
            self.logger.error(f"❌ Clustering failed: {e}")
            return []

    def _extract_clustering_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """Extract features for clustering from price and volume data."""
        try:
            features = []

            if self.config.use_price_features and price_data is not None:
                # Price-based features
                price_features = self._extract_price_features(price_data)
                features.append(price_features)

            if self.config.use_volume_features and volume_data is not None:
                # Volume-based features
                volume_features = self._extract_volume_features(volume_data)
                features.append(volume_features)

            if self.config.use_time_features:
                # Time-based features
                time_features = self._extract_time_features(price_data.index)
                features.append(time_features)

            if not features:
                self.logger.error("No features extracted")
                return None

            # Combine all features
            combined_features = pd.concat(features, axis=1)

            # Normalize features
            normalized_features = self._normalize_features(combined_features)

            return normalized_features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _extract_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract price-based features for clustering."""
        features = pd.DataFrame(index=price_data.index)

        # Basic price features
        features['price'] = price_data['close']
        features['price_change'] = price_data['close'].pct_change()
        features['price_volatility'] = price_data['close'].rolling(20).std()

        # Price patterns
        features['local_max'] = (price_data['high'] == price_data['high'].rolling(5, center=True).max()).astype(int)
        features['local_min'] = (price_data['low'] == price_data['low'].rolling(5, center=True).min()).astype(int)

        # Support/resistance indicators
        features['resistance_level'] = (price_data['high'] > price_data['high'].rolling(10).max().shift(1)).astype(int)
        features['support_level'] = (price_data['low'] < price_data['low'].rolling(10).min().shift(1)).astype(int)

        return features.fillna(0)

    def _extract_volume_features(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-based features for clustering."""
        features = pd.DataFrame(index=volume_data.index)

        # Volume features
        features['volume'] = volume_data['volume']
        features['volume_change'] = volume_data['volume'].pct_change()
        features['volume_spike'] = (volume_data['volume'] > volume_data['volume'].rolling(20).mean() * 2).astype(int)

        return features.fillna(0)

    def _extract_time_features(self, index: pd.Index) -> pd.DataFrame:
        """Extract time-based features for clustering."""
        features = pd.DataFrame(index=index)

        # Time-based features
        features['hour'] = index.hour
        features['day_of_week'] = index.dayofweek
        features['day_of_month'] = index.day
        features['month'] = index.month

        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        return features

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for clustering."""
        if self.config.feature_normalization == 'standard':
            scaler = StandardScaler()
            normalized = pd.DataFrame(
                scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
        elif self.config.feature_normalization == 'minmax':
            normalized = (features - features.min()) / (features.max() - features.min())
        else:  # robust or none
            normalized = features.copy()

        return normalized.fillna(0)

    def _perform_clustering(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Perform clustering on features."""
        try:
            if self.config.clustering_method == 'dbscan':
                clusterer = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
                labels = clusterer.fit_predict(features)

            elif self.config.clustering_method == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=self.config.n_clusters)
                labels = clusterer.fit_predict(features)

            else:  # hybrid approach
                # Try DBSCAN first, fall back to agglomerative
                try:
                    clusterer = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
                    labels = clusterer.fit_predict(features)
                    # If DBSCAN creates too many clusters or mostly noise, use agglomerative
                    n_clusters_db = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters_db > self.config.n_clusters * 2 or n_clusters_db < 2:
                        raise ValueError("DBSCAN clustering unsatisfactory")
                except:
                    clusterer = AgglomerativeClustering(n_clusters=self.config.n_clusters)
                    labels = clusterer.fit_predict(features)

            # Calculate silhouette score if possible
            unique_labels = set(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(features) - 1:
                try:
                    self.silhouette_score = silhouette_score(features, labels)
                except:
                    self.silhouette_score = None
            else:
                self.silhouette_score = None

            return labels

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return None

    def _create_cluster_results(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        index: pd.Index
    ) -> List[ClusterResult]:
        """Create cluster results from clustering output."""
        cluster_results = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue

            # Get indices for this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_size = len(cluster_indices)

            if cluster_size < self.config.min_samples:
                continue

            # Calculate cluster centroid (mean price)
            cluster_feature_data = features.iloc[cluster_indices]
            centroid_price = cluster_feature_data['price'].mean() if 'price' in cluster_feature_data.columns else 0.0

            # Calculate silhouette score for this cluster if available
            silhouette_score_val = None
            if self.silhouette_score is not None and len(unique_labels) > 1:
                try:
                    # Calculate silhouette for this specific cluster
                    from sklearn.metrics import silhouette_samples
                    sample_scores = silhouette_samples(features, labels)
                    cluster_scores = sample_scores[cluster_indices]
                    silhouette_score_val = np.mean(cluster_scores)
                except:
                    pass

            # Create cluster result
            cluster_result = ClusterResult(
                cluster_id=int(label),
                level_indices=cluster_indices.tolist(),
                centroid_price=float(centroid_price),
                cluster_size=cluster_size,
                silhouette_score=float(silhouette_score_val) if silhouette_score_val is not None else 0.0,
                confidence=min(1.0, cluster_size / 50.0)  # Confidence based on cluster size
            )

            cluster_results.append(cluster_result)

        return cluster_results

    async def _validate_clusters_with_backtesting(
        self,
        cluster_results: List[ClusterResult],
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> List[ClusterResult]:
        """Validate clusters using backtesting."""
        if not self.backtesting_engine:
            return cluster_results

        validated_results = []

        for cluster_result in cluster_results:
            try:
                # Create SR levels from cluster
                sr_levels = self._create_sr_levels_from_cluster(
                    cluster_result, price_data, volume_data
                )

                if not sr_levels:
                    continue

                # Run backtest for this cluster
                backtest_result = await self.backtesting_engine.backtest_sr_strategy(
                    market_data=price_data,
                    sr_levels=sr_levels,
                    strategy_name=f"Cluster_{cluster_result.cluster_id}"
                )

                # Update cluster result with backtest score
                cluster_result.backtest_score = backtest_result.sharpe_ratio

                # Only include clusters that meet minimum backtest criteria
                if backtest_result.success and backtest_result.sharpe_ratio >= self.config.min_backtest_score:
                    validated_results.append(cluster_result)

            except Exception as e:
                self.logger.warning(f"Backtest validation failed for cluster {cluster_result.cluster_id}: {e}")
                # Include cluster even if backtest fails (with None score)
                validated_results.append(cluster_result)

        return validated_results

    def _create_sr_levels_from_cluster(
        self,
        cluster_result: ClusterResult,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> List[SRLevel]:
        """Create SR levels from a cluster result."""
        sr_levels = []

        try:
            # Get price data for cluster indices
            cluster_indices = cluster_result.level_indices
            if not cluster_indices:
                return sr_levels

            cluster_prices = price_data.iloc[cluster_indices]

            # Create SR level at cluster centroid
            level = SRLevel(
                price=cluster_result.centroid_price,
                strength=min(1.0, cluster_result.confidence),
                touches=cluster_result.cluster_size,
                level_type='support' if cluster_result.centroid_price < price_data['close'].mean() else 'resistance',
                start_time=price_data.index[min(cluster_indices)],
                end_time=price_data.index[max(cluster_indices)],
                confidence=cluster_result.confidence
            )

            sr_levels.append(level)

        except Exception as e:
            self.logger.error(f"Failed to create SR levels from cluster: {e}")

        return sr_levels


def get_backtesting_enhanced_clustering(
    config: Optional[BacktestingEnhancedConfig] = None
) -> BacktestingEnhancedClustering:
    """
    Factory function to create backtesting-enhanced clustering.

    Args:
        config: Clustering configuration (creates default if None)

    Returns:
        Configured BacktestingEnhancedClustering instance
    """
    if config is None:
        config = BacktestingEnhancedConfig()

    return BacktestingEnhancedClustering(config)


# Export main classes and functions
__all__ = [
    'BacktestingEnhancedClustering',
    'BacktestingEnhancedConfig',
    'ClusterResult',
    'get_backtesting_enhanced_clustering'
]
