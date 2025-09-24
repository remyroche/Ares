"""
NAS Clustering Component

Integrates NAS-based clustering capabilities into the market analysis pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

logger = logging.getLogger(__name__)


class NASClusteringComponent(BaseMarketAnalysisComponent):
    """
    NAS Clustering Component

    Performs optimal regime clustering using NAS-based features and
    advanced clustering algorithms with economic significance evaluation.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize NAS clustering component."""
        super().__init__(config)

        # Initialize clustering components
        self._initialize_clustering_components()

        logger.info("✅ NAS Clustering Component initialized")

    def _initialize_clustering_components(self):
        """Initialize the clustering components."""
        try:
            from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_clustering_integration import (
                EnhancedNASClusteringIntegration,
                NASClusteringConfig
            )

            # Create clustering configuration from component config
            clustering_config = NASClusteringConfig(
                clustering_algorithm=self.config.custom_params.get('clustering_algorithm', 'auto'),
                n_clusters=self.config.custom_params.get('n_clusters', 8),
                max_features=self.config.custom_params.get('max_features', 50),
                min_cluster_size=self.config.custom_params.get('min_cluster_size', 10),
                economic_evaluation=self.config.custom_params.get('economic_evaluation', True),
                stability_threshold=self.config.custom_params.get('stability_threshold', 0.6)
            )

            self.clustering_integration = EnhancedNASClusteringIntegration(clustering_config)
            logger.info("✅ NAS clustering integration initialized successfully")

        except ImportError as e:
            logger.warning(f"NAS clustering integration not available: {e}, using fallback")
            self.clustering_integration = None

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['optimal_regime_clustering_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS clustering.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with clustering results
        """
        try:
            if not isinstance(data, pd.DataFrame):
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Input data must be a pandas DataFrame"
                )

            # Get regime assignments from pipeline state
            regime_assignments = pipeline_state.get('regime_assignments', {})

            if not regime_assignments:
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="No regime assignments found in pipeline state"
                )

            regime_labels = regime_assignments.get('regime_labels')
            if regime_labels is None or len(regime_labels) == 0:
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="No valid regime labels found"
                )

            # Execute NAS clustering
            if self.clustering_integration is not None:
                clustering_result = self.clustering_integration.perform_clustering(
                    data, regime_labels
                )

                if not clustering_result.get('success', False):
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message="NAS clustering failed"
                    )

                # Create comprehensive artifact
                artifact = {
                    'optimal_regime_clustering_result': {
                        'nas_clusters': clustering_result.get('clusters', {}),
                        'nas_clustering_metrics': clustering_result.get('clustering_metrics', {}),
                        'cluster_assignments': clustering_result.get('cluster_assignments', []),
                        'cluster_characteristics': clustering_result.get('cluster_characteristics', {}),
                        'execution_time': clustering_result.get('execution_time', 0.0),
                        'metadata': {
                            'method': 'enhanced_nas_clustering',
                            'n_clusters': len(set(clustering_result.get('cluster_assignments', []))),
                            'algorithm': clustering_result.get('algorithm', 'unknown'),
                            'success': True
                        }
                    }
                }

                return ComponentResult(
                    success=True,
                    artifacts=artifact,
                    metadata={'component_type': 'nas_clustering'}
                )
            else:
                # Fallback implementation
                return self._fallback_clustering(data, regime_labels)

        except Exception as e:
            logger.error(f"NAS clustering failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )

    def _fallback_clustering(self, data: pd.DataFrame, regime_labels: np.ndarray) -> ComponentResult:
        """Fallback clustering using basic methods."""
        try:
            logger.info("🔄 Using fallback NAS clustering")

            # Create basic clustering based on regime labels
            unique_regimes = np.unique(regime_labels)
            n_clusters = len(unique_regimes)

            # Create cluster assignments (same as regime labels for fallback)
            cluster_assignments = regime_labels.copy()

            # Calculate basic clustering metrics
            silhouette_score = self._calculate_silhouette_score(data, cluster_assignments)
            calinski_harabasz_score = self._calculate_calinski_harabasz_score(data, cluster_assignments)

            # Create cluster characteristics
            cluster_characteristics = {}
            for i, regime in enumerate(unique_regimes):
                mask = regime_labels == regime
                cluster_data = data[mask]

                if len(cluster_data) > 0:
                    cluster_characteristics[i] = {
                        'size': len(cluster_data),
                        'mean_return': cluster_data['close'].pct_change().mean(),
                        'volatility': cluster_data['close'].pct_change().std(),
                        'regime_id': int(regime),
                        'data_points': len(cluster_data)
                    }

            # Create fallback artifact
            artifact = {
                'optimal_regime_clustering_result': {
                    'nas_clusters': cluster_characteristics,
                    'nas_clustering_metrics': {
                        'silhouette_score': silhouette_score,
                        'calinski_harabasz_score': calinski_harabasz_score,
                        'davies_bouldin_score': 1 - silhouette_score,  # Approximation
                        'n_clusters': n_clusters,
                        'total_samples': len(data),
                        'cluster_sizes': [len(cluster_data) for cluster_data in [data[regime_labels == r] for r in unique_regimes]]
                    },
                    'cluster_assignments': cluster_assignments,
                    'cluster_characteristics': cluster_characteristics,
                    'execution_time': 0.0,
                    'metadata': {
                        'method': 'fallback',
                        'n_clusters': n_clusters,
                        'algorithm': 'basic',
                        'success': True
                    }
                }
            }

            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'nas_clustering'}
            )

        except Exception as e:
            logger.error(f"Fallback clustering failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Fallback clustering failed: {e}"
            )

    def _calculate_silhouette_score(self, data: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering evaluation."""
        try:
            from sklearn.metrics import silhouette_score

            # Create feature matrix from OHLCV data
            features = data[['open', 'high', 'low', 'close', 'volume']].fillna(0).values

            # Calculate silhouette score
            if len(np.unique(labels)) > 1 and len(features) > len(np.unique(labels)):
                score = silhouette_score(features, labels)
                return max(0, min(1, score))  # Clamp to [0, 1]
            else:
                return 0.5  # Default score
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            return 0.5

    def _calculate_calinski_harabasz_score(self, data: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score for clustering evaluation."""
        try:
            from sklearn.metrics import calinski_harabasz_score

            # Create feature matrix from OHLCV data
            features = data[['open', 'high', 'low', 'close', 'volume']].fillna(0).values

            # Calculate Calinski-Harabasz score
            if len(np.unique(labels)) > 1 and len(features) > len(np.unique(labels)):
                score = calinski_harabasz_score(features, labels)
                return max(0, min(1000, score)) / 1000  # Normalize to [0, 1]
            else:
                return 0.5  # Default score
        except Exception as e:
            logger.warning(f"Calinski-Harabasz score calculation failed: {e}")
            return 0.5