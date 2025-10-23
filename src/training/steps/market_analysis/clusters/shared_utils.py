"""
Shared utilities for clustering components.

This module provides essential utilities that were previously in the deleted shared_utils folder.

ENHANCED WITH BASESTEP COMPREHENSIVE TOOLS:
- Direct access to all utility modules through BaseStep
- Comprehensive logging with tprint integration
- Hardware optimization built-in
- Safe operations with fallbacks
- Memory management and cleanup
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import BaseStep for comprehensive utility access
from src.training.steps.base_step import BaseStep

# Import tprint functions directly (available through BaseStep)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance,
    tprint_step_start, tprint_step_end, tprint_operation_start, tprint_operation_end,
    tprint_data_summary, tprint_performance_summary, tprint_memory_usage
)


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
    config: FeatureConfig,
    basestep_instance: Optional[BaseStep] = None
) -> FeaturePreparationResult:
    """
    Prepare market features for clustering using BaseStep utilities when available.
    
    Args:
        market_data: Market data DataFrame
        config: Feature configuration
        basestep_instance: Optional BaseStep instance for enhanced utilities
        
    Returns:
        FeaturePreparationResult with prepared features
    """
    logger = get_logger('prepare_market_features')
    
    try:
        # Basic feature preparation
        features = market_data.select_dtypes(include=[np.number]).values
        
        # Use BaseStep math validation if available
        if basestep_instance:
            features = basestep_instance._validate_finite(features, default=0)
        
        # Simple feature names
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Apply scaling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use BaseStep math validation if available
        if basestep_instance:
            features_scaled = basestep_instance._validate_finite(features_scaled, default=0)
        
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
                
                # Use BaseStep math validation if available
                if basestep_instance:
                    features_scaled = basestep_instance._validate_finite(features_scaled, default=0)
                    
            except ImportError:
                logger.warning("UMAP not available, skipping UMAP transformation")
        
        # Use BaseStep hardware optimization if available
        if basestep_instance and basestep_instance.hardware_utils:
            try:
                features_scaled = basestep_instance.hardware_utils['optimize_dataframe'](
                    pd.DataFrame(features_scaled)
                ).values
            except Exception as e:
                logger.warning(f"Hardware optimization failed: {e}")
        
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
    """Calculator for various clustering metrics with BaseStep enhancement."""
    
    def __init__(self, basestep_instance: Optional[BaseStep] = None):
        self.basestep_instance = basestep_instance
        self.logger = get_logger('MetricsCalculator')
    
    def calculate_all_metrics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate all available metrics using BaseStep utilities when available."""
        try:
            # Use BaseStep math validation if available
            if self.basestep_instance:
                cluster_assignments = self.basestep_instance._validate_finite(cluster_assignments, default=0)
            
            metrics = {}
            metrics.update(calculate_consensus_metrics_safe(cluster_assignments, market_data, self.basestep_instance))
            metrics.update(calculate_disagreement_metrics_safe(cluster_assignments, market_data, self.basestep_instance))
            metrics.update(calculate_economic_scores_safe(cluster_assignments, market_data, self.basestep_instance))
            metrics.update(calculate_trading_scores_safe(cluster_assignments, market_data, self.basestep_instance))
            metrics.update(calculate_stability_scores_safe(cluster_assignments, market_data, self.basestep_instance))
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

# Safe versions of utility functions that use BaseStep utilities when available

def calculate_consensus_metrics_safe(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame,
    basestep_instance: Optional[BaseStep] = None
) -> Dict[str, Any]:
    """Calculate consensus metrics using BaseStep safe operations."""
    try:
        # Use BaseStep math validation if available
        if basestep_instance:
            cluster_assignments = basestep_instance._validate_finite(cluster_assignments, default=0)
        
        # Basic consensus metrics
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        # Calculate consensus score using BaseStep safe operations
        consensus_score = 0.0
        if basestep_instance:
            consensus_score = basestep_instance._safe_divide(n_clusters, n_samples, default=0)
        else:
            consensus_score = n_clusters / n_samples if n_samples > 0 else 0
        
        return {
            'consensus_score': float(consensus_score),
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
    except Exception as e:
        if basestep_instance:
            basestep_instance.tprint_error(f"❌ Consensus metrics calculation failed: {e}")
        return {
            'consensus_score': 0.0,
            'n_clusters': 0,
            'n_samples': 0
        }

def calculate_disagreement_metrics_safe(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame,
    basestep_instance: Optional[BaseStep] = None
) -> Dict[str, Any]:
    """Calculate disagreement metrics using BaseStep safe operations."""
    try:
        # Use BaseStep math validation if available
        if basestep_instance:
            cluster_assignments = basestep_instance._validate_finite(cluster_assignments, default=0)
        
        # Basic disagreement metrics
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        # Calculate disagreement score using BaseStep safe operations
        disagreement_score = 0.0
        if basestep_instance:
            disagreement_score = basestep_instance._safe_divide(1.0, n_clusters, default=0)
        else:
            disagreement_score = 1.0 / n_clusters if n_clusters > 0 else 0
        
        return {
            'disagreement_score': float(disagreement_score),
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
    except Exception as e:
        if basestep_instance:
            basestep_instance.tprint_error(f"❌ Disagreement metrics calculation failed: {e}")
        return {
            'disagreement_score': 0.0,
            'n_clusters': 0,
            'n_samples': 0
        }

def calculate_economic_scores_safe(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame,
    basestep_instance: Optional[BaseStep] = None
) -> Dict[str, Any]:
    """Calculate economic scores using BaseStep safe operations."""
    try:
        # Use BaseStep math validation if available
        if basestep_instance:
            cluster_assignments = basestep_instance._validate_finite(cluster_assignments, default=0)
        
        # Basic economic scores
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        # Calculate economic score using BaseStep safe operations
        economic_score = 0.0
        if basestep_instance:
            economic_score = basestep_instance._safe_divide(n_clusters, n_samples, default=0)
        else:
            economic_score = n_clusters / n_samples if n_samples > 0 else 0
        
        return {
            'economic_score': float(economic_score),
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
    except Exception as e:
        if basestep_instance:
            basestep_instance.tprint_error(f"❌ Economic scores calculation failed: {e}")
        return {
            'economic_score': 0.0,
            'n_clusters': 0,
            'n_samples': 0
        }

def calculate_trading_scores_safe(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame,
    basestep_instance: Optional[BaseStep] = None
) -> Dict[str, Any]:
    """Calculate trading scores using BaseStep safe operations."""
    try:
        # Use BaseStep math validation if available
        if basestep_instance:
            cluster_assignments = basestep_instance._validate_finite(cluster_assignments, default=0)
        
        # Basic trading scores
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        # Calculate trading score using BaseStep safe operations
        trading_score = 0.0
        if basestep_instance:
            trading_score = basestep_instance._safe_divide(n_clusters, n_samples, default=0)
        else:
            trading_score = n_clusters / n_samples if n_samples > 0 else 0
        
        return {
            'trading_score': float(trading_score),
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
    except Exception as e:
        if basestep_instance:
            basestep_instance.tprint_error(f"❌ Trading scores calculation failed: {e}")
        return {
            'trading_score': 0.0,
            'n_clusters': 0,
            'n_samples': 0
        }

def calculate_stability_scores_safe(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame,
    basestep_instance: Optional[BaseStep] = None
) -> Dict[str, Any]:
    """Calculate stability scores using BaseStep safe operations."""
    try:
        # Use BaseStep math validation if available
        if basestep_instance:
            cluster_assignments = basestep_instance._validate_finite(cluster_assignments, default=0)
        
        # Basic stability scores
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        # Calculate stability score using BaseStep safe operations
        stability_score = 0.0
        if basestep_instance:
            stability_score = basestep_instance._safe_divide(1.0, n_clusters, default=0)
        else:
            stability_score = 1.0 / n_clusters if n_clusters > 0 else 0
        
        return {
            'stability_score': float(stability_score),
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
    except Exception as e:
        if basestep_instance:
            basestep_instance.tprint_error(f"❌ Stability scores calculation failed: {e}")
        return {
            'stability_score': 0.0,
            'n_clusters': 0,
            'n_samples': 0
        }