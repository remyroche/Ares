"""
Enhanced HDP-HMM Clustering Integration

This module provides comprehensive HDP-HMM clustering integration that combines
existing feature bank features with HDP-HMM-specific preprocessing for optimal
Bayesian nonparametric regime discovery.

Target: 50-100 comprehensive features optimized for HDP-HMM clustering
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_structured, tprint_timer
)

# Import feature bank integration
from .feature_bank_integration import (
    FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory
)

# Import HDP-HMM clusterer
from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
    HDPHMMClusterer, HDPHMMConfig, HDPHMMResult, HMM_AVAILABLE
)


class EnhancedHDPHMMClusteringIntegration:
    """
    Enhanced HDP-HMM Clustering Integration.
    
    Provides comprehensive features optimized for Bayesian nonparametric clustering
    using Sticky HDP-HMM models. Features are selected to capture temporal dependencies
    and regime-dependent dynamics.
    """
    
    def __init__(self,
                 min_features: int = 50,
                 max_features: int = 100,
                 enable_comprehensive_features: bool = True,
                 enable_pca_reduction: bool = True,
                 pca_components: int = 10,
                 alpha: float = 3.0,
                 kappa: float = 50.0,
                 gamma: float = 3.0,
                 n_iterations: int = 100,
                 max_states: int = 20):
        """
        Initialize Enhanced HDP-HMM Clustering Integration.
        
        Args:
            min_features: Minimum number of features
            max_features: Maximum number of features
            enable_comprehensive_features: Enable comprehensive feature generation
            enable_pca_reduction: Enable PCA dimensionality reduction
            pca_components: Number of PCA components
            alpha: HDP-HMM concentration parameter (higher = more regimes)
            kappa: HDP-HMM stickiness parameter (higher = longer durations)
            gamma: HDP-HMM base distribution hyperparameter
            n_iterations: Number of Gibbs sampling iterations
            max_states: Maximum number of states to consider
        """
        tprint_info("🚀 Initializing Enhanced HDP-HMM Clustering Integration")
        
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_pca_reduction = enable_pca_reduction
        self.pca_components = pca_components
        
        # HDP-HMM parameters
        self.alpha = alpha
        self.kappa = kappa
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.max_states = max_states
        
        # Log configuration
        tprint_structured({
            "min_features": min_features,
            "max_features": max_features,
            "enable_comprehensive_features": enable_comprehensive_features,
            "enable_pca_reduction": enable_pca_reduction,
            "pca_components": pca_components,
            "alpha": alpha,
            "kappa": kappa,
            "gamma": gamma,
            "n_iterations": n_iterations,
            "max_states": max_states
        }, level="INFO")
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            tprint_info("🔧 Configuring Feature Bank Integrator for HDP-HMM clustering")
            
            config = FeatureBankConfig()
            config.hdbscan_min_features = min_features
            config.hdbscan_max_features = max_features
            # Weight features for temporal and regime-dependent patterns
            config.hdbscan_weights = {
                FeatureBankCategory.VOLATILITY: 0.3,   # Volatility regime changes
                FeatureBankCategory.TREND: 0.25,       # Trend dynamics
                FeatureBankCategory.MOMENTUM: 0.2,     # Momentum shifts
                FeatureBankCategory.VOLUME: 0.15,      # Volume patterns
                FeatureBankCategory.CLUSTERING: 0.1    # Auxiliary clustering features
            }
            
            self.feature_integrator = FeatureBankIntegrator(config)
            tprint_success("✅ Feature Bank Integrator initialized")
        else:
            tprint_warning("⚠️ Comprehensive features disabled")
            self.feature_integrator = None
    
    def get_comprehensive_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for HDP-HMM clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        tprint_info("🔍 Generating comprehensive features for HDP-HMM clustering")
        
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            if self.enable_comprehensive_features:
                result = self.feature_integrator.get_comprehensive_features_for_task(
                    'hdbscan_clustering', data
                )
                
                result.update({
                    'clustering_method': 'hdp_hmm',
                    'temporal_aware': True
                })
                
                return result
            else:
                return self._get_basic_features(data)
    
    def _get_basic_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to basic features."""
        return {
            'features': {},
            'feature_names': [],
            'feature_count': 0,
            'clustering_method': 'hdp_hmm'
        }
    
    def prepare_data_for_clustering(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for HDP-HMM clustering.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names, metadata)
        """
        tprint_info("🔧 Preparing data for HDP-HMM clustering")
        
        with tprint_timer("Data Preparation", level="PERFORMANCE"):
            # Get comprehensive features
            feature_result = self.get_comprehensive_clustering_features(data)
            features = feature_result['features']
            feature_names = feature_result['feature_names']
            
            if not features:
                tprint_warning("⚠️ No features generated")
                return np.array([]).reshape(len(data), 0), [], feature_result
            
            # Convert to numpy array
            feature_matrix = np.column_stack([features[name] for name in feature_names])
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            metadata = feature_result.copy()
            metadata.update({
                'final_shape': feature_matrix.shape,
                'clustering_method': 'hdp_hmm'
            })
            
            tprint_success(f"✅ Data preparation completed: {feature_matrix.shape}")
            
            return feature_matrix, feature_names, metadata
    
    def cluster_with_hdp_hmm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform HDP-HMM clustering.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing clustering results
        """
        tprint_info("🎯 Starting HDP-HMM Clustering")
        
        if not HMM_AVAILABLE:
            tprint_error("❌ HMM libraries not available")
            raise ImportError("HMM libraries not available")
        
        with tprint_timer("HDP-HMM Clustering", level="PERFORMANCE"):
            # Prepare data
            feature_matrix, feature_names, metadata = self.prepare_data_for_clustering(data)
            
            if feature_matrix.size == 0:
                raise ValueError("No features available for clustering")
            
            # Create HDP-HMM configuration
            hdp_config = HDPHMMConfig(
                alpha=self.alpha,
                kappa=self.kappa,
                gamma=self.gamma,
                n_iterations=self.n_iterations,
                max_states=self.max_states,
                enable_pca=self.enable_pca_reduction,
                pca_components=self.pca_components
            )
            
            # Create clusterer and fit
            clusterer = HDPHMMClusterer(hdp_config)
            result = clusterer.fit_predict(feature_matrix)
            
            tprint_success(f"🎉 HDP-HMM clustering completed: {result.n_clusters} regimes")
        
        return {
            'cluster_labels': result.cluster_labels,
            'cluster_probabilities': result.cluster_probabilities,
            'n_clusters': result.n_clusters,
            'transition_matrix': result.transition_matrix,
            'emission_params': result.emission_params,
            'state_durations': result.state_durations,
            'feature_names': feature_names,
            'feature_matrix': feature_matrix,
            'clusterer': clusterer,
            'metadata': metadata,
            'quality_metrics': {
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score,
                'log_likelihood': result.log_likelihood,
                'posterior_mean_states': result.posterior_mean_states,
                'posterior_std_states': result.posterior_std_states,
                'transition_persistence': result.transition_persistence
            },
            'hdp_result': result
        }


# Convenience function
def perform_enhanced_hdp_hmm_clustering(
    data: pd.DataFrame,
    min_features: int = 50,
    max_features: int = 100,
    alpha: float = 3.0,
    kappa: float = 50.0,
    gamma: float = 3.0,
    n_iterations: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform enhanced HDP-HMM clustering.
    
    Args:
        data: Market data DataFrame
        min_features: Minimum number of features
        max_features: Maximum number of features
        alpha: Concentration parameter (higher = more regimes)
        kappa: Stickiness parameter (higher = longer durations)
        gamma: Base distribution hyperparameter
        n_iterations: Number of Gibbs sampling iterations
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with clustering results
    """
    integrator = EnhancedHDPHMMClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        alpha=alpha,
        kappa=kappa,
        gamma=gamma,
        n_iterations=n_iterations,
        **kwargs
    )
    
    return integrator.cluster_with_hdp_hmm(data)


__all__ = [
    'EnhancedHDPHMMClusteringIntegration',
    'perform_enhanced_hdp_hmm_clustering'
]
