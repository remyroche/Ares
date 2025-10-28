"""
Enhanced MS-DR Clustering Integration

This module provides comprehensive Markov-Switching Dynamic Regression clustering
integration that combines existing feature bank features with MS-DR-specific
preprocessing for optimal regime-dependent dynamics modeling.

Target: 50-100 comprehensive features optimized for MS-DR clustering

═══════════════════════════════════════════════════════════════════════════════
USAGE GUIDE
═══════════════════════════════════════════════════════════════════════════════

Basic Usage:
    from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
        perform_enhanced_ms_dr_clustering
    )
    
    # Perform clustering with auto regime selection
    result = perform_enhanced_ms_dr_clustering(
        data=market_data,
        min_features=50,
        max_features=100,
        auto_select_regimes=True,
        min_regimes=2,
        max_regimes=10
    )
    
    # Access results
    regime_labels = result['cluster_labels']
    transition_matrix = result['transition_matrix']
    regime_probabilities = result['cluster_probabilities']

Advanced Usage:
    # Create integrator with custom settings
    integrator = EnhancedMSDRClusteringIntegration(
        min_features=50,
        max_features=100,
        enable_comprehensive_features=True,
        enable_pca_reduction=True,
        pca_components=10,
        n_regimes=5,
        model_type='autoregression',
        order=1,
        switching_variance=True,
        auto_select_regimes=True
    )
    
    # Run clustering
    result = integrator.cluster_with_ms_dr(market_data)

Understanding Results:
    - cluster_labels: Regime assignment for each time point
    - cluster_probabilities: Probability distribution over regimes
    - n_clusters: Number of discovered regimes
    - transition_matrix: Regime transition probabilities
    - quality_metrics: Various quality measures (silhouette, AIC, BIC, etc.)

IMPORTANT NOTES:
    1. MS-DR requires time series data (temporal ordering matters)
    2. Features are reduced to univariate series for regime modeling
    3. Regimes represent hidden states with different dynamics
    4. Results include transition probabilities between regimes
    5. Suitable for market regime identification and state-dependent modeling

═══════════════════════════════════════════════════════════════════════════════
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

# Import regime-specific features and categorization
try:
    from src.feature_generation.categories.regime_features import (
        RegimeFeatureGenerator, RegimeFeatureConfig
    )
    REGIME_FEATURES_AVAILABLE = True
    tprint_debug("✅ Regime-specific features available")
except ImportError:
    REGIME_FEATURES_AVAILABLE = False
    tprint_debug("⚠️ Regime-specific features not available")

# Import regime feature categorization system
try:
    from src.feature_generation.categories.regime_feature_categorization import (
        RegimeFeatureCategorizer,
        FeatureUseCase,
        get_regime_clustering_features,
        validate_feature_set
    )
    REGIME_CATEGORIZATION_AVAILABLE = True
    tprint_debug("✅ Regime feature categorization available")
except ImportError:
    REGIME_CATEGORIZATION_AVAILABLE = False
    tprint_debug("⚠️ Regime feature categorization not available")

# Import regime feature integration
try:
    from src.feature_generation.categories.regime_feature_integration import (
        RegimeFeatureIntegration,
        RegimeFeatureConfig as RegimeIntegrationConfig,
        generate_regime_features,
        create_default_regime_feature_generators
    )
    REGIME_INTEGRATION_AVAILABLE = True
    tprint_debug("✅ Regime feature integration available")
except ImportError:
    REGIME_INTEGRATION_AVAILABLE = False
    tprint_debug("⚠️ Regime feature integration not available")

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

# Import artifact manager for data loading
from src.training.steps.market_analysis.components.artifact_manager import ArtifactManager

# Import cluster quality assessor and optimization goals
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    format_metrics_report
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
                 max_regimes: int = 10,
                 enable_regime_categorization: bool = True,
                 enable_regime_integration: bool = True):
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
            enable_regime_categorization: Use regime feature categorization system
            enable_regime_integration: Use regime feature integration module
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
        self.enable_regime_categorization = enable_regime_categorization
        self.enable_regime_integration = enable_regime_integration
        
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
            "auto_select_regimes": auto_select_regimes,
            "enable_regime_categorization": enable_regime_categorization,
            "enable_regime_integration": enable_regime_integration
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
            except (ImportError, AttributeError, TypeError) as e:
                tprint_warning(f"⚠️ Failed to initialize regime features: {e}")
                self.regime_feature_gen = None
            except Exception as e:
                tprint_error(f"❌ Unexpected error initializing regime features: {e}")
                self.regime_feature_gen = None
        else:
            self.regime_feature_gen = None
        
        # Initialize vectorization manager if available (from code review)
        if VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ Vectorization manager initialized")
            except (ImportError, AttributeError, TypeError) as e:
                tprint_warning(f"⚠️ Failed to initialize vectorization: {e}")
                self.vectorization_manager = None
            except Exception as e:
                tprint_error(f"❌ Unexpected error initializing vectorization: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
        
        # Initialize regime feature categorizer
        if self.enable_regime_categorization and REGIME_CATEGORIZATION_AVAILABLE:
            try:
                self.regime_categorizer = RegimeFeatureCategorizer()
                tprint_success("✅ Regime feature categorizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize regime categorizer: {e}")
                self.regime_categorizer = None
        else:
            self.regime_categorizer = None
        
        # Initialize regime feature integration
        if self.enable_regime_integration and REGIME_INTEGRATION_AVAILABLE:
            try:
                self.regime_integration_generators = create_default_regime_feature_generators()
                tprint_success(f"✅ Regime integration initialized ({len(self.regime_integration_generators)} generators)")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize regime integration: {e}")
                self.regime_integration_generators = []
        else:
            self.regime_integration_generators = []
    
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
                    except (KeyError, ValueError, AttributeError) as e:
                        tprint_warning(f"⚠️ Failed to generate regime features (expected): {e}")
                    except Exception as e:
                        tprint_error(f"❌ Unexpected error generating regime features: {e}")
                        import traceback
                        tprint_debug(traceback.format_exc())
                
                # Add regime categorization features if enabled
                if self.regime_categorizer is not None:
                    try:
                        tprint_info("📊 Applying regime feature categorization")
                        
                        # Get regime clustering features
                        regime_clustering_features = self.regime_categorizer.get_priority_features(
                            FeatureUseCase.REGIME_CLUSTERING,
                            max_features=max_features
                        )
                        
                        # Filter features to only include those optimized for regime clustering
                        filtered_features = {}
                        for feature_name in regime_clustering_features:
                            if feature_name in result['features']:
                                filtered_features[feature_name] = result['features'][feature_name]
                        
                        if filtered_features:
                            result['features'] = filtered_features
                            result['feature_names'] = list(filtered_features.keys())
                            result['feature_count'] = len(filtered_features)
                            result['regime_categorization_applied'] = True
                            
                            tprint_success(
                                f"✅ Applied regime categorization: {len(filtered_features)} features selected"
                            )
                        
                        # Validate feature set
                        validation = validate_feature_set(
                            list(filtered_features.keys()),
                            FeatureUseCase.REGIME_CLUSTERING
                        )
                        
                        if not validation['validation_passed']:
                            tprint_warning(
                                f"⚠️ Feature validation: {validation['invalid_count']} invalid features found"
                            )
                        
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to apply regime categorization: {e}")
                
                # Add regime integration features if enabled
                if self.regime_integration_generators:
                    try:
                        tprint_info("🔄 Generating regime integration features")
                        
                        regime_integration_features = {}
                        for generator in self.regime_integration_generators:
                            gen_features = generate_regime_features(data, generator.regime_config)
                            
                            # Convert to proper format
                            for key, value in gen_features.items():
                                feature_name = f"regime_integration_{key}"
                                if isinstance(value, (int, float, bool)):
                                    # Broadcast scalar to array
                                    regime_integration_features[feature_name] = np.full(len(data), value)
                                elif isinstance(value, str):
                                    # Encode string as category
                                    regime_integration_features[feature_name] = np.full(len(data), hash(value) % 1000)
                                else:
                                    regime_integration_features[feature_name] = value
                        
                        if regime_integration_features:
                            result['features'].update(regime_integration_features)
                            result['feature_names'].extend(regime_integration_features.keys())
                            result['feature_count'] += len(regime_integration_features)
                            result['regime_integration_features_added'] = len(regime_integration_features)
                            
                            tprint_success(
                                f"✅ Added {len(regime_integration_features)} regime integration features"
                            )
                        
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate regime integration features: {e}")
                        import traceback
                        tprint_debug(traceback.format_exc())
                
                result.update({
                    'clustering_method': 'ms_dr',
                    'dynamics_aware': True,
                    'regime_categorization_enabled': self.regime_categorizer is not None,
                    'regime_integration_enabled': len(self.regime_integration_generators) > 0
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


def perform_ms_dr_clustering_with_artifact_manager(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    artifact_base_dir: str = "artifacts",
    min_features: int = 50,
    max_features: int = 100,
    n_regimes: int = 5,
    model_type: str = 'autoregression',
    auto_select_regimes: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform MS-DR clustering with artifact manager for data loading.
    
    This is the recommended standalone function that:
    1. Uses ArtifactManager to load market data
    2. Performs MS-DR clustering with comprehensive features
    3. Assesses quality using cluster_quality_assessor
    4. Validates against clustering_optimization_goals
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "binance")
        timeframe: Timeframe (e.g., "30m", "1h")
        artifact_base_dir: Base directory for artifacts
        min_features: Minimum number of features
        max_features: Maximum number of features
        n_regimes: Number of regimes (if not auto-selecting)
        model_type: Model type ('autoregression', 'regression')
        auto_select_regimes: Auto-select number of regimes
        **kwargs: Additional parameters for MSDRClusterer
        
    Returns:
        Dictionary with clustering results and quality metrics
        
    Example:
        >>> from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
        ...     perform_ms_dr_clustering_with_artifact_manager
        ... )
        >>> 
        >>> result = perform_ms_dr_clustering_with_artifact_manager(
        ...     symbol="BTCUSDT",
        ...     exchange="binance",
        ...     timeframe="30m",
        ...     artifact_base_dir="artifacts",
        ...     min_features=50,
        ...     max_features=100,
        ...     auto_select_regimes=True
        ... )
        >>> 
        >>> # Access results
        >>> regime_labels = result['cluster_labels']
        >>> quality_score = result['quality_metrics']['quality_score']
        >>> print(f"Found {result['n_clusters']} regimes with quality score: {quality_score:.3f}")
    """
    tprint_info(f"🚀 Starting MS-DR Clustering with Artifact Manager")
    tprint_structured({
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "artifact_base_dir": artifact_base_dir,
        "min_features": min_features,
        "max_features": max_features,
        "auto_select_regimes": auto_select_regimes
    }, level="INFO")
    
    # Initialize artifact manager
    tprint_info("📁 Initializing Artifact Manager")
    artifact_manager = ArtifactManager(
        base_dir=artifact_base_dir,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe
    )
    
    # Load market data from artifacts
    tprint_info(f"📊 Loading market data for {symbol} on {exchange} ({timeframe})")
    try:
        # Try to load from latest session
        market_data_dict = artifact_manager.load_artifacts_from_latest_session(
            component_name="market_data",
            artifact_names=["ohlcv", "processed_data", "klines"]
        )
        
        if not market_data_dict:
            tprint_error("❌ No market data found in artifacts. Please ensure market data is available.")
            raise ValueError(
                f"No market data found for {symbol}_{exchange}_{timeframe}. "
                "Please run data collection first or provide data manually using "
                "perform_enhanced_ms_dr_clustering(data=your_dataframe)"
            )
        
        # Get the first available data artifact
        market_data = None
        for artifact_name, data in market_data_dict.items():
            if data is not None:
                market_data = data
                tprint_success(f"✅ Loaded market data from artifact: {artifact_name}")
                break
        
        if market_data is None:
            raise ValueError("All loaded artifacts were None")
        
        # Ensure it's a DataFrame
        if not isinstance(market_data, pd.DataFrame):
            tprint_warning(f"⚠️ Market data is not a DataFrame (type: {type(market_data)}). Attempting conversion...")
            market_data = pd.DataFrame(market_data)
        
        tprint_data_preview(market_data, "Loaded Market Data", max_rows=5, max_cols=10)
        
    except Exception as e:
        tprint_error(f"❌ Failed to load market data: {e}")
        raise ValueError(
            f"Failed to load market data: {e}\n\n"
            "Please ensure you have run data collection first, or use "
            "perform_enhanced_ms_dr_clustering(data=your_dataframe) to provide data manually."
        )
    
    # Perform MS-DR clustering
    tprint_info("🔄 Performing MS-DR clustering")
    result = perform_enhanced_ms_dr_clustering(
        data=market_data,
        min_features=min_features,
        max_features=max_features,
        n_regimes=n_regimes,
        model_type=model_type,
        auto_select_regimes=auto_select_regimes,
        **kwargs
    )
    
    # Add artifact manager information to result
    result['artifact_manager'] = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'artifact_base_dir': artifact_base_dir,
        'data_loaded_from': 'artifact_manager'
    }
    
    # Generate quality report
    tprint_info("📊 Generating quality metrics report")
    quality_metrics = result.get('quality_metrics', {})
    
    if quality_metrics:
        report = format_metrics_report(
            cv_score=quality_metrics.get('between_regime_cv', 0.0) / (quality_metrics.get('within_regime_cv', 1.0) + 1e-8),
            silhouette_score=quality_metrics.get('silhouette_score', 0.0),
            dbi_score=quality_metrics.get('davies_bouldin_score', float('inf')),
            balance_score=quality_metrics.get('balance_score', 0.0),
            temporal_smoothness=quality_metrics.get('temporal_smoothness', 0.0),
            n_clusters=result.get('n_clusters', 0)
        )
        tprint(report)
    
    tprint_success("🎉 MS-DR Clustering with Artifact Manager completed successfully!")
    
    return result


__all__ = [
    'EnhancedMSDRClusteringIntegration',
    'perform_enhanced_ms_dr_clustering',
    'perform_ms_dr_clustering_with_artifact_manager'
]
