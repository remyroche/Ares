"""
Shared utilities for clustering components.

This module provides essential utilities that were previously in the deleted shared_utils folder.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


@dataclass
class FeatureConfig:
    """Configuration for feature preparation."""
    n_features: int = 50
    use_pca: bool = True
    pca_components: int = 20
    use_umap: bool = False
    umap_components: int = 10
    scaler_type: str = 'robust'


@dataclass
class FeaturePreparationResult:
    """Result from feature preparation."""
    features: np.ndarray
    feature_names: List[str]
    scaler: Any
    pca: Optional[Any] = None
    umap: Optional[Any] = None
    feature_scores: Dict[str, float] = None


def prepare_market_features(
    market_data: pd.DataFrame,
    config: FeatureConfig
) -> FeaturePreparationResult:
    """
    Prepare market features for clustering.
    
    Args:
        market_data: Market data DataFrame
        config: Feature configuration
        
    Returns:
        FeaturePreparationResult with prepared features
    """
    logger = get_logger('prepare_market_features')
    
    try:
        # Basic feature preparation
        features = market_data.select_dtypes(include=[np.number]).values
        
        # Simple feature names
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Apply scaling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA if requested
        pca = None
        if config.use_pca and config.pca_components < features_scaled.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=config.pca_components)
            features_scaled = pca.fit_transform(features_scaled)
            feature_names = [f"pca_{i}" for i in range(config.pca_components)]
        
        # Apply UMAP if requested
        umap = None
        if config.use_umap:
            try:
                import umap
                umap = umap.UMAP(n_components=config.umap_components)
                features_scaled = umap.fit_transform(features_scaled)
                feature_names = [f"umap_{i}" for i in range(config.umap_components)]
            except ImportError:
                logger.warning("UMAP not available, skipping UMAP transformation")
        
        return FeaturePreparationResult(
            features=features_scaled,
            feature_names=feature_names,
            scaler=scaler,
            pca=pca,
            umap=umap,
            feature_scores={name: 1.0 for name in feature_names}
        )
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        raise


def calculate_consensus_metrics(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate consensus metrics for clustering results."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        return {
            'n_clusters': n_clusters,
            'n_samples': n_samples,
            'consensus_score': 0.8,  # Placeholder
            'stability_score': 0.7   # Placeholder
        }
    except Exception:
        return {'n_clusters': 0, 'n_samples': 0, 'consensus_score': 0.0, 'stability_score': 0.0}


def calculate_disagreement_metrics(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate disagreement metrics for clustering results."""
    try:
        return {
            'disagreement_score': 0.2,  # Placeholder
            'uncertainty_score': 0.3   # Placeholder
        }
    except Exception:
        return {'disagreement_score': 0.0, 'uncertainty_score': 0.0}


def calculate_economic_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate economic scores for clustering results."""
    try:
        return {
            'economic_score': 0.6,  # Placeholder
            'trading_score': 0.5    # Placeholder
        }
    except Exception:
        return {'economic_score': 0.0, 'trading_score': 0.0}


def calculate_trading_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate trading scores for clustering results."""
    try:
        return {
            'trading_score': 0.5,  # Placeholder
            'profitability_score': 0.4  # Placeholder
        }
    except Exception:
        return {'trading_score': 0.0, 'profitability_score': 0.0}


def calculate_stability_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate stability scores for clustering results."""
    try:
        return {
            'stability_score': 0.7,  # Placeholder
            'consistency_score': 0.6  # Placeholder
        }
    except Exception:
        return {'stability_score': 0.0, 'consistency_score': 0.0}


class MetricsCalculator:
    """Calculator for various clustering metrics."""
    
    def __init__(self):
        self.logger = get_logger('MetricsCalculator')
    
    def calculate_all_metrics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate all available metrics."""
        try:
            metrics = {}
            metrics.update(calculate_consensus_metrics(cluster_assignments, market_data))
            metrics.update(calculate_disagreement_metrics(cluster_assignments, market_data))
            metrics.update(calculate_economic_scores(cluster_assignments, market_data))
            metrics.update(calculate_trading_scores(cluster_assignments, market_data))
            metrics.update(calculate_stability_scores(cluster_assignments, market_data))
            return metrics
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {}


class CharacteristicsGenerator:
    """Generator for cluster characteristics."""
    
    def __init__(self):
        self.logger = get_logger('CharacteristicsGenerator')
    
    def generate_characteristics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate cluster characteristics."""
        try:
            n_clusters = len(np.unique(cluster_assignments))
            characteristics = {
                'n_clusters': n_clusters,
                'cluster_sizes': [np.sum(cluster_assignments == i) for i in range(n_clusters)],
                'cluster_characteristics': {}
            }
            
            for i in range(n_clusters):
                cluster_mask = cluster_assignments == i
                cluster_data = market_data[cluster_mask]
                
                characteristics['cluster_characteristics'][f'cluster_{i}'] = {
                    'size': np.sum(cluster_mask),
                    'mean_volume': cluster_data['volume'].mean() if 'volume' in cluster_data.columns else 0,
                    'mean_price': cluster_data['close'].mean() if 'close' in cluster_data.columns else 0
                }
            
            return characteristics
        except Exception as e:
            self.logger.error(f"Characteristics generation failed: {e}")
            return {'n_clusters': 0, 'cluster_sizes': [], 'cluster_characteristics': {}}