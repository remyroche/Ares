"""
Enhanced MS-DR Clustering Integration

This module provides comprehensive Markov-Switching Dynamic Regression clustering
integration that combines existing feature bank features with MS-DR-specific
preprocessing for optimal regime-dependent dynamics modeling.

Target: 50-100 comprehensive features optimized for MS-DR clustering
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

# Import regime-specific features (from code review)
try:
    from src.feature_generation.categories.regime_features import (
        RegimeFeatureGenerator, RegimeFeatureConfig
    )
    REGIME_FEATURES_AVAILABLE = True
    tprint_debug("✅ Regime-specific features available")
except ImportError:
    REGIME_FEATURES_AVAILABLE = False
    tprint_debug("⚠️ Regime-specific features not available")

# Import optimization utilities (from code review)
try:
    from src.utils.ml_common.optimization.hpo_utils import get_hpo_optimizer
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    tprint_debug("⚠️ HPO utilities not available")

try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False
    tprint_debug("⚠️ Unified vectorization not available")

# Import MS-DR clusterer
from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import (
    MSDRClusterer, MSDRConfig, MSDRResult, MS_AVAILABLE
)


class EnhancedMSDRClusteringIntegration:
    """
    Enhanced MS-DR Clustering Integration.
    
    Provides comprehensive features optimized for Markov-Switching Dynamic Regression
    clustering. Features are selected to capture regime-dependent dynamics and
    heteroskedasticity patterns.
    """
    
    def __init__(self,
                 min_features: int = 50,
                 max_features: int = 100,
                 enable_comprehensive_features: bool = True,
                 enable_pca_reduction: bool = True,
                 pca_components: int = 10,
                 n_regimes: int = 5,
                 model_type: str = 'autoregression',
                 order: int = 1,
                 switching_variance: bool = True,
                 auto_select_regimes: bool = True,
                 min_regimes: int = 2,
                 max_regimes: int = 10):
        """
        Initialize Enhanced MS-DR Clustering Integration.
        
        Args:
            min_features: Minimum number of features
            max_features: Maximum number of features
            enable_comprehensive_features: Enable comprehensive feature generation
            enable_pca_reduction: Enable PCA dimensionality reduction
            pca_components: Number of PCA components
            n_regimes: Number of regimes (if not auto-selecting)
            model_type: Model type ('autoregression', 'regression')
            order: Autoregression order
            switching_variance: Allow variance to switch across regimes
            auto_select_regimes: Auto-select number of regimes using IC
            min_regimes: Minimum number of regimes to consider
            max_regimes: Maximum number of regimes to consider
        """
        tprint_info("🚀 Initializing Enhanced MS-DR Clustering Integration")
        
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_pca_reduction = enable_pca_reduction
        self.pca_components = pca_components
        
        # MS-DR parameters
        self.n_regimes = n_regimes
        self.model_type = model_type
        self.order = order
        self.switching_variance = switching_variance
        self.auto_select_regimes = auto_select_regimes
        self.min_regimes = min_regimes
        self.max_regimes = max_regimes
        
        # Log configuration
        tprint_structured({
            "min_features": min_features,
            "max_features": max_features,
            "enable_comprehensive_features": enable_comprehensive_features,
            "enable_pca_reduction": enable_pca_reduction,
            "pca_components": pca_components,
            "n_regimes": n_regimes,
            "model_type": model_type,
            "order": order,
            "switching_variance": switching_variance,
            "auto_select_regimes": auto_select_regimes
        }, level="INFO")
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            tprint_info("🔧 Configuring Feature Bank Integrator for MS-DR clustering")
            
            config = FeatureBankConfig()
            config.hdbscan_min_features = min_features
            config.hdbscan_max_features = max_features
            # Weight features for regime-dependent dynamics
            config.hdbscan_weights = {
                FeatureBankCategory.VOLATILITY: 0.35,  # Switching variance
                FeatureBankCategory.TREND: 0.3,        # Regime-dependent trends
                FeatureBankCategory.MOMENTUM: 0.2,     # Dynamic shifts
                FeatureBankCategory.VOLUME: 0.1,       # Volume regimes
                FeatureBankCategory.CLUSTERING: 0.05   # Auxiliary features
            }
            
            self.feature_integrator = FeatureBankIntegrator(config)
            tprint_success("✅ Feature Bank Integrator initialized")
        else:
            tprint_warning("⚠️ Comprehensive features disabled")
            self.feature_integrator = None
        
        # Initialize regime feature generator if available (from code review)
        if REGIME_FEATURES_AVAILABLE:
            try:
                regime_config = RegimeFeatureConfig()
                self.regime_feature_gen = RegimeFeatureGenerator(regime_config)
                tprint_success("✅ Regime feature generator initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize regime features: {e}")
                self.regime_feature_gen = None
        else:
            self.regime_feature_gen = None
        
        # Initialize vectorization manager if available (from code review)
        if VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ Vectorization manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize vectorization: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
    
    def get_comprehensive_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for MS-DR clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        tprint_info("🔍 Generating comprehensive features for MS-DR clustering")
        tprint_data_preview(data, "Input Market Data", max_rows=3, max_cols=5)
        
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            if self.enable_comprehensive_features:
                # Get base features from feature bank
                result = self.feature_integrator.get_comprehensive_features_for_task(
                    'hdbscan_clustering', data
                )
                
                # Add regime-specific features if available (from code review)
                if self.regime_feature_gen is not None:
                    try:
                        tprint_info("📊 Generating regime-specific features")
                        regime_features = self.regime_feature_gen.generate_features(data)
                        
                        # Merge regime features with base features
                        if regime_features and 'features' in regime_features:
                            result['features'].update(regime_features['features'])
                            result['feature_names'].extend(regime_features.get('feature_names', []))
                            result['regime_features_added'] = len(regime_features.get('feature_names', []))
                            
                            tprint_success(
                                f"✅ Added {result['regime_features_added']} regime-specific features"
                            )
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate regime features: {e}")
                
                result.update({
                    'clustering_method': 'ms_dr',
                    'dynamics_aware': True
                })
                
                tprint_data_format(result['features'], "Generated Features", check_compatibility=True)
                return result
            else:
                return self._get_basic_features(data)
    
    def _get_basic_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to basic features."""
        return {
            'features': {},
            'feature_names': [],
            'feature_count': 0,
            'clustering_method': 'ms_dr'
        }
    
    def prepare_data_for_clustering(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for MS-DR clustering.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names, metadata)
        """
        tprint_info("🔧 Preparing data for MS-DR clustering")
        
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
                'clustering_method': 'ms_dr'
            })
            
            tprint_success(f"✅ Data preparation completed: {feature_matrix.shape}")
            
            return feature_matrix, feature_names, metadata
    
    def cluster_with_ms_dr(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform MS-DR clustering.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing clustering results
        """
        tprint_info("🎯 Starting MS-DR Clustering")
        
        if not MS_AVAILABLE:
            tprint_error("❌ MS-DR libraries not available")
            raise ImportError("statsmodels.tsa.regime_switching not available")
        
        with tprint_timer("MS-DR Clustering", level="PERFORMANCE"):
            # Prepare data
            feature_matrix, feature_names, metadata = self.prepare_data_for_clustering(data)
            
            if feature_matrix.size == 0:
                raise ValueError("No features available for clustering")
            
            # Create MS-DR configuration
            ms_config = MSDRConfig(
                n_regimes=self.n_regimes,
                model_type=self.model_type,
                order=self.order,
                switching_variance=self.switching_variance,
                auto_select_regimes=self.auto_select_regimes,
                min_regimes=self.min_regimes,
                max_regimes=self.max_regimes,
                enable_pca=self.enable_pca_reduction,
                pca_components=self.pca_components
            )
            
            # Create clusterer and fit
            clusterer = MSDRClusterer(ms_config)
            result = clusterer.fit_predict(feature_matrix)
            
            tprint_success(f"🎉 MS-DR clustering completed: {result.n_clusters} regimes")
        
        return {
            'cluster_labels': result.cluster_labels,
            'cluster_probabilities': result.cluster_probabilities,
            'n_clusters': result.n_clusters,
            'transition_matrix': result.transition_matrix,
            'regime_params': result.regime_params,
            'regime_variances': result.regime_variances,
            'regime_durations': result.regime_durations,
            'feature_names': feature_names,
            'feature_matrix': feature_matrix,
            'clusterer': clusterer,
            'metadata': metadata,
            'quality_metrics': {
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score,
                'log_likelihood': result.log_likelihood,
                'aic': result.aic,
                'bic': result.bic,
                'hqic': result.hqic,
                'transition_persistence': result.transition_persistence
            },
            'ms_result': result
        }


# Convenience function
def perform_enhanced_ms_dr_clustering(
    data: pd.DataFrame,
    min_features: int = 50,
    max_features: int = 100,
    n_regimes: int = 5,
    model_type: str = 'autoregression',
    auto_select_regimes: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform enhanced MS-DR clustering.
    
    Args:
        data: Market data DataFrame
        min_features: Minimum number of features
        max_features: Maximum number of features
        n_regimes: Number of regimes (if not auto-selecting)
        model_type: Model type ('autoregression', 'regression')
        auto_select_regimes: Auto-select number of regimes
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with clustering results
    """
    integrator = EnhancedMSDRClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        n_regimes=n_regimes,
        model_type=model_type,
        auto_select_regimes=auto_select_regimes,
        **kwargs
    )
    
    return integrator.cluster_with_ms_dr(data)


__all__ = [
    'EnhancedMSDRClusteringIntegration',
    'perform_enhanced_ms_dr_clustering'
]
