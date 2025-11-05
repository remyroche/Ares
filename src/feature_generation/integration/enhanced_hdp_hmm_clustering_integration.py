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
    tprint_debug, tprint_performance, tprint_structured, tprint_timer,
    tprint_data_preview, tprint_data_format
)

# Import feature bank integration
from .feature_bank_integration import (
    FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory
)

# Define availability constants first to avoid redefinition warnings
REGIME_FEATURES_AVAILABLE = False
REGIME_CATEGORIZATION_AVAILABLE = False
REGIME_INTEGRATION_AVAILABLE = False
HPO_AVAILABLE = False
VECTORIZATION_AVAILABLE = False

# Import regime-specific features
# NOTE: RegimeFeatureGenerator is an optional enhancement
# The system works fine without it using base feature bank features
try:
    from src.feature_generation.categories.regime_features import (
        RegimeFeatureGenerator, RegimeFeatureConfig
    )
    REGIME_FEATURES_AVAILABLE = True
    tprint_debug("✅ Regime-specific features available")
except ImportError as e:
    tprint_debug(
        f"ℹ️ Regime-specific features not available (optional): {e}. "
        "Using base feature bank features only."
    )

# Import regime feature categorization for intelligent feature selection
try:
    from src.feature_generation.categories.regime_feature_categorization import (
        RegimeFeatureCategorizer,
        FeatureUseCase,
        get_regime_clustering_features,
        get_hdbscan_features,
        validate_feature_set
    )
    REGIME_CATEGORIZATION_AVAILABLE = True
    tprint_debug("✅ Regime feature categorization available")
except ImportError as e:
    tprint_debug(f"ℹ️ Regime feature categorization not available: {e}")

# Import regime feature integration for regime-aware features
try:
    from src.feature_generation.categories.regime_feature_integration import (
        RegimeFeatureIntegration,
        RegimeFeatureConfig as RegimeIntegrationConfig,
        generate_regime_features
    )
    REGIME_INTEGRATION_AVAILABLE = True
    tprint_debug("✅ Regime feature integration available")
except ImportError as e:
    tprint_debug(f"ℹ️ Regime feature integration not available: {e}")

# Import optimization utilities (from code review)
try:
    from src.utils.ml_common.optimization.hpo_utils import get_hpo_optimizer
    HPO_AVAILABLE = True
except ImportError:
    tprint_debug("⚠️ HPO utilities not available")

try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORIZATION_AVAILABLE = True
except ImportError:
    tprint_debug("⚠️ Unified vectorization not available")

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
                 max_states: int = 20,
                 use_regime_categorization: bool = True,
                 use_regime_integration: bool = True):
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
            use_regime_categorization: Use intelligent feature categorization (default: True)
            use_regime_integration: Use regime-aware feature integration (default: True)
        """
        tprint_info("🚀 Initializing Enhanced HDP-HMM Clustering Integration")
        
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_pca_reduction = enable_pca_reduction
        self.pca_components = pca_components
        self.use_regime_categorization = use_regime_categorization and REGIME_CATEGORIZATION_AVAILABLE
        self.use_regime_integration = use_regime_integration and REGIME_INTEGRATION_AVAILABLE
        
        # HDP-HMM parameters
        self.alpha = alpha
        self.kappa = kappa
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.max_states = max_states
        
        # Initialize regime feature categorizer
        self.regime_categorizer = None
        if self.use_regime_categorization:
            try:
                self.regime_categorizer = RegimeFeatureCategorizer()
                tprint_success("✅ Regime feature categorizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize regime categorizer: {e}")
                self.use_regime_categorization = False
        
        # Initialize regime feature integration
        self.regime_integrator = None
        if self.use_regime_integration:
            try:
                regime_int_config = RegimeIntegrationConfig(
                    enable_regime_detection=True,
                    enable_adaptive_features=True,
                    enable_regime_transitions=True,
                    lookback_period=20
                )
                self.regime_integrator = RegimeFeatureIntegration(regime_int_config)
                tprint_success("✅ Regime feature integration initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize regime integration: {e}")
                self.use_regime_integration = False
        
        # Log configuration
        tprint_structured({
            "min_features": min_features,
            "max_features": max_features,
            "enable_comprehensive_features": enable_comprehensive_features,
            "enable_pca_reduction": enable_pca_reduction,
            "pca_components": pca_components,
            "use_regime_categorization": self.use_regime_categorization,
            "use_regime_integration": self.use_regime_integration,
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
            # NOTE: Using hdbscan config parameters as general clustering configuration
            # These parameters work for both HDBSCAN and HDP-HMM clustering
            config.hdbscan_min_features = min_features
            config.hdbscan_max_features = max_features
            # Weight features for temporal and regime-dependent patterns
            # These weights emphasize features that capture regime transitions
            config.hdbscan_weights = {
                FeatureBankCategory.VOLATILITY: 0.3,   # Volatility regime changes (high priority)
                FeatureBankCategory.TREND: 0.25,       # Trend dynamics (important for regimes)
                FeatureBankCategory.MOMENTUM: 0.2,     # Momentum shifts (regime indicators)
                FeatureBankCategory.VOLUME: 0.15,      # Volume patterns (regime confirmation)
                FeatureBankCategory.CLUSTERING: 0.1    # Auxiliary clustering features
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
                from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_success("✅ Vectorization manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize vectorization: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
    
    def get_comprehensive_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for HDP-HMM clustering.
        Uses intelligent feature categorization and regime-aware integration.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        tprint_info("🔍 Generating comprehensive features for HDP-HMM clustering")
        tprint_data_preview(data, "Input Market Data", max_rows=3, max_cols=5)
        
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            if self.enable_comprehensive_features:
                # Strategy 1: Use regime categorization for intelligent feature selection
                if self.use_regime_categorization and self.regime_categorizer:
                    tprint_info("🎯 Using intelligent regime feature categorization")
                    result = self._get_categorized_features(data)
                else:
                    # Fallback: Get base features from feature bank
                    # NOTE: Using 'hdbscan_clustering' task which provides general clustering features
                    # (volatility, trend, momentum, volume) that are also appropriate for HDP-HMM.
                    tprint_info("📊 Using feature bank integration")
                    result = self.feature_integrator.get_comprehensive_features_for_task(
                        'hdbscan_clustering', data
                    )
                
                # Strategy 2: Add regime-aware integration features
                if self.use_regime_integration and self.regime_integrator:
                    try:
                        tprint_info("🔄 Adding regime-aware integration features")
                        regime_int_features = self.regime_integrator._generate_regime_features(data)
                        
                        # Add to feature set
                        if regime_int_features:
                            for feat_name, feat_value in regime_int_features.items():
                                # Convert categorical to numerical if needed
                                if isinstance(feat_value, str):
                                    # Skip categorical features for now
                                    continue
                                # Create feature array
                                feat_array = np.full(len(data), feat_value, dtype=np.float64)
                                result['features'][f'regime_int_{feat_name}'] = feat_array
                                result['feature_names'].append(f'regime_int_{feat_name}')
                            
                            result['regime_integration_added'] = len(regime_int_features)
                            tprint_success(
                                f"✅ Added {result['regime_integration_added']} regime integration features"
                            )
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to add regime integration features: {e}")
                
                # Strategy 3: Add regime-specific features (original implementation)
                if self.regime_feature_gen is not None:
                    try:
                        tprint_info("📈 Generating regime-specific features")
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
                    'clustering_method': 'hdp_hmm',
                    'temporal_aware': True,
                    'regime_categorization_used': self.use_regime_categorization,
                    'regime_integration_used': self.use_regime_integration
                })
                
                tprint_data_format(result['features'], "Generated Features", check_compatibility=True)
                return result
            else:
                return self._get_basic_features(data)
    
    def _get_categorized_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get features using intelligent regime categorization.
        
        This uses RegimeFeatureCategorizer to select the most appropriate features
        for regime clustering, ranked by priority and suitability.
        """
        try:
            # Get priority features for regime clustering
            # This intelligently selects features optimized for regime discovery
            priority_features = self.regime_categorizer.get_priority_features(
                FeatureUseCase.REGIME_CLUSTERING,
                max_features=self.max_features
            )
            
            tprint_info(f"🎯 Selected {len(priority_features)} priority features for regime clustering")
            
            # Get feature requirements
            requirements = self.regime_categorizer.get_feature_requirements(
                FeatureUseCase.REGIME_CLUSTERING
            )
            
            tprint_structured({
                "total_categories": requirements['total_categories'],
                "total_features": requirements['total_features'],
                "stability_required": requirements['stability_required'],
                "lookahead_safe": requirements['lookahead_safe'],
                "categories": requirements['categories']
            }, level="INFO")
            
            # Generate features using feature bank (filtered by categorization)
            result = self.feature_integrator.get_comprehensive_features_for_task(
                'hdbscan_clustering', data
            )
            
            # Filter to only priority features (if they exist in generated features)
            if result['feature_names']:
                # Keep features that are in priority list
                filtered_features = {}
                filtered_names = []
                
                for feat_name in result['feature_names']:
                    # Check if feature matches any priority feature pattern
                    if any(priority_feat in feat_name for priority_feat in priority_features):
                        if feat_name in result['features']:
                            filtered_features[feat_name] = result['features'][feat_name]
                            filtered_names.append(feat_name)
                
                # If we have enough filtered features, use them
                if len(filtered_names) >= self.min_features:
                    result['features'] = filtered_features
                    result['feature_names'] = filtered_names
                    result['categorization_applied'] = True
                    tprint_success(f"✅ Filtered to {len(filtered_names)} categorized features")
                else:
                    # Keep original features if filtering removes too many
                    result['categorization_applied'] = False
                    tprint_warning(f"⚠️ Filtering left too few features ({len(filtered_names)}), using all")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to use categorization: {e}")
            # Fallback to standard feature bank
            return self.feature_integrator.get_comprehensive_features_for_task(
                'hdbscan_clustering', data
            )
    
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
            
            # Handle NaN and inf values with proper imputation
            n_nan = np.isnan(feature_matrix).sum()
            n_inf = np.isinf(feature_matrix).sum()
            
            if n_nan > 0 or n_inf > 0:
                nan_ratio = n_nan / feature_matrix.size
                inf_ratio = n_inf / feature_matrix.size
                
                tprint_warning(
                    f"⚠️ Cleaning feature matrix: {n_nan} NaN ({nan_ratio:.2%}) "
                    f"and {n_inf} inf ({inf_ratio:.2%}) values"
                )
                
                # Use median imputation for NaN values
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median', copy=False)
                
                try:
                    feature_matrix = imputer.fit_transform(feature_matrix)
                    tprint_info("   ✅ Applied median imputation for NaN values")
                except Exception as e:
                    tprint_warning(f"   ⚠️ Median imputation failed: {e}, using zero fill")
                    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
                
                # Clip extreme values (inf becomes large but bounded)
                feature_matrix = np.clip(feature_matrix, -1e3, 1e3)
                tprint_info("   ✅ Clipped extreme values to [-1000, 1000]")
            
            metadata = feature_result.copy()
            metadata.update({
                'final_shape': feature_matrix.shape,
                'clustering_method': 'hdp_hmm',
                'nan_values_cleaned': int(n_nan),
                'inf_values_cleaned': int(n_inf)
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
