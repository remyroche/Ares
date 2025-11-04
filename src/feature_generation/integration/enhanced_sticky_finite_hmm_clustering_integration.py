"""
Enhanced Sticky Finite HMM Clustering Integration

This module provides comprehensive Sticky Finite HMM clustering integration that combines
existing feature bank features with preprocessing for optimal Bayesian regime discovery.

Target: 50-100 comprehensive features optimized for Sticky Finite HMM clustering
Pattern: Mirrors enhanced_hdp_hmm_clustering_integration.py but for fixed-K model
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

# Import regime-specific features (optional)
try:
    from src.feature_generation.categories.regime_features import (
        RegimeFeatureGenerator, RegimeFeatureConfig
    )
    REGIME_FEATURES_AVAILABLE = True
    tprint_debug("✅ Regime-specific features available")
except ImportError as e:
    REGIME_FEATURES_AVAILABLE = False
    tprint_debug(
        f"ℹ️ Regime-specific features not available (optional): {e}. "
        "Using base feature bank features only."
    )

# Import regime feature categorization (optional)
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
    REGIME_CATEGORIZATION_AVAILABLE = False
    tprint_debug(f"ℹ️ Regime feature categorization not available: {e}")

# Import regime feature integration (optional)
try:
    from .regime_feature_integration import (
        RegimeFeatureIntegration
    )
    REGIME_INTEGRATION_AVAILABLE = True
    tprint_debug("✅ Regime feature integration available")
except ImportError as e:
    REGIME_INTEGRATION_AVAILABLE = False
    tprint_debug(f"ℹ️ Regime feature integration not available: {e}")

# Import Sticky Finite HMM components
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
    StickyFiniteHMMClusterer, StickyFiniteHMMConfig, StickyFiniteHMMResult,
    DEPENDENCIES_AVAILABLE
)


class EnhancedStickyFiniteHMMClusteringIntegration:
    """
    Enhanced Sticky Finite HMM Clustering Integration.
    
    Provides 50-100 comprehensive features optimized for Sticky Finite HMM regime discovery
    by combining existing feature bank features with clustering-specific preprocessing.
    
    Pipeline: Feature Generation → Feature Selection (50-100) → PCA (10) → Sticky Finite HMM
    
    Key Differences from HDP-HMM:
    - Fixed K=5 states (not nonparametric)
    - Uses VB/SVI instead of Gibbs sampling
    - Faster convergence with KMeans initialization
    """
    
    def __init__(self,
                 min_features: int = 50,
                 max_features: int = 100,
                 enable_comprehensive_features: bool = True,
                 enable_pca_reduction: bool = True,
                 pca_components: int = 15,  # Use 15 components (same as core model, can use up to 20)
                 K: int = 5,
                 n_mixtures: int = 1,  # Number of Gaussian mixtures per state
                 base_alpha: float = 0.5,
                 kappa: float = 10.0,
                 num_iters: int = 800,
                 lr: float = 1e-2,
                 use_regime_categorization: bool = True,
                 use_regime_integration: bool = True,
                 enable_mtf_features: bool = True,  # Enable multi-timeframe regime features
                 mtf_timeframes: Optional[List[str]] = None):
        """
        Initialize Enhanced Sticky Finite HMM Clustering Integration.
        
        Args:
            min_features: Minimum number of features to use
            max_features: Maximum number of features to use
            enable_comprehensive_features: Enable comprehensive feature generation
            enable_pca_reduction: Enable PCA reduction
            pca_components: Number of PCA components
            K: Number of states (regimes) - fixed
            base_alpha: Concentration for off-diagonal transitions
            kappa: Stickiness parameter
            num_iters: Number of SVI iterations
            lr: Learning rate
            use_regime_categorization: Use intelligent feature categorization
            use_regime_integration: Use regime-aware feature integration
        """
        tprint_info("🚀 Initializing Enhanced Sticky Finite HMM Clustering Integration")
        
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_pca_reduction = enable_pca_reduction
        self.pca_components = pca_components
        
        # Sticky Finite HMM parameters
        self.K = K
        self.n_mixtures = n_mixtures
        self.base_alpha = base_alpha
        self.kappa = kappa
        self.num_iters = num_iters
        self.lr = lr
        
        self.use_regime_categorization = use_regime_categorization
        self.use_regime_integration = use_regime_integration
        
        # Multi-timeframe features
        self.enable_mtf_features = enable_mtf_features
        self.mtf_timeframes = mtf_timeframes or ['4h', '1d']  # Default: 4h and daily context
        
        # Log configuration
        tprint_structured({
            "min_features": min_features,
            "max_features": max_features,
            "enable_comprehensive_features": enable_comprehensive_features,
            "enable_pca_reduction": enable_pca_reduction,
            "pca_components": pca_components,
            "K": K,
            "n_mixtures": n_mixtures,
            "base_alpha": base_alpha,
            "kappa": kappa,
            "num_iters": num_iters,
            "lr": lr,
            "enable_mtf_features": enable_mtf_features,
            "mtf_timeframes": self.mtf_timeframes
        }, level="INFO")
        
        # Initialize feature bank integrator
        # NOTE: Use SAME configuration as HDP-HMM to ensure identical feature generation
        if enable_comprehensive_features:
            tprint_info("🔧 Configuring Feature Bank Integrator (identical to HDP-HMM)")
            
            feature_config = FeatureBankConfig()
            # Use HDP-HMM compatible parameters with microstructure addition
            feature_config.hdbscan_min_features = min_features
            feature_config.hdbscan_max_features = max_features
            # Weight features for temporal and regime-dependent patterns
            # Enhanced with microstructure for better regime detection
            feature_config.hdbscan_weights = {
                FeatureBankCategory.VOLATILITY: 0.30,      # Volatility regime changes (high priority)
                FeatureBankCategory.TREND: 0.25,           # Trend dynamics (important for regimes)
                FeatureBankCategory.MOMENTUM: 0.20,        # Momentum shifts (regime indicators)
                FeatureBankCategory.VOLUME: 0.12,          # Volume patterns (regime confirmation)
                FeatureBankCategory.MICROSTRUCTURE: 0.08,  # Microstructure for regime granularity
                FeatureBankCategory.CLUSTERING: 0.05       # Auxiliary clustering features
            }
            
            # Enable quantile-based volatility regime differentiation
            # Uses 3-regime classification (low/medium/high) based on volatility quantiles (0.33, 0.67)
            feature_config.enable_volatility_regime_quantiles = True
            feature_config.volatility_quantile_thresholds = [0.33, 0.67]  # Low, Medium, High regimes
            feature_config.volatility_regime_windows = [10, 20, 50]  # Multiple lookback windows
            
            self.feature_integrator = FeatureBankIntegrator(config=feature_config)
            tprint_success("✅ Feature bank integrator initialized (HDP-HMM compatible)")
        else:
            self.feature_integrator = None
        
        # Initialize regime categorizer (optional)
        if use_regime_categorization and REGIME_CATEGORIZATION_AVAILABLE:
            try:
                self.regime_categorizer = RegimeFeatureCategorizer()
                tprint_success("✅ Regime feature categorizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Could not initialize regime categorizer: {e}")
                self.regime_categorizer = None
        else:
            self.regime_categorizer = None
        
        # Initialize regime integration (optional)
        if use_regime_integration and REGIME_INTEGRATION_AVAILABLE:
            try:
                self.regime_integration = RegimeFeatureIntegration()
                tprint_success("✅ Regime feature integration initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Could not initialize regime integration: {e}")
                self.regime_integration = None
        else:
            self.regime_integration = None
    
    def get_comprehensive_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for Sticky Finite HMM clustering.
        Uses intelligent feature categorization and regime-aware integration.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        tprint_info("🔍 Generating comprehensive features for Sticky Finite HMM clustering")
        tprint_data_preview(data, "Input Market Data", max_rows=3, max_cols=5)
        
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            if self.enable_comprehensive_features:
                # Strategy 1: Use regime categorization for intelligent feature selection
                if self.use_regime_categorization and self.regime_categorizer:
                    tprint_info("🎯 Using intelligent regime feature categorization")
                    result = self._get_categorized_features(data)
                else:
                    # Fallback: Get base features from feature bank
                    tprint_info("📊 Using feature bank integration")
                    from src.feature_generation.integration.feature_bank_integration import MLTask
                    result = self.feature_integrator.get_comprehensive_features_for_task(
                        MLTask.HDBSCAN_CLUSTERING, data
                    )
                
                # Strategy 2: Add regime-aware integration features
                if self.use_regime_integration and self.regime_integration:
                    tprint_info("🔧 Adding regime-aware integration features")
                    regime_features = self.regime_integration.generate_regime_features(data)
                    
                    # Merge regime features
                    result['features'] = pd.concat(
                        [result['features'], regime_features],
                        axis=1
                    )
                    result['feature_names'] = result['feature_names'] + regime_features.columns.tolist()
                    
                    tprint_info(f"✅ Added {len(regime_features.columns)} regime-aware features")
                
                # Strategy 3: Add multi-timeframe regime context features
                if self.enable_mtf_features:
                    tprint_info("🌐 Adding multi-timeframe (MTF) regime context features")
                    mtf_features = self._generate_mtf_regime_features(data)
                    
                    if mtf_features is not None and len(mtf_features.columns) > 0:
                        # Merge MTF features
                        result['features'] = pd.concat(
                            [result['features'], mtf_features],
                            axis=1
                        )
                        result['feature_names'] = result['feature_names'] + mtf_features.columns.tolist()
                        
                        tprint_success(f"✅ Added {len(mtf_features.columns)} multi-timeframe features")
                
                tprint_success(f"✅ Generated {len(result.get('feature_names', []))} total features")
                
            else:
                # Simple mode: just use basic features
                tprint_warning("⚠️ Comprehensive features disabled, using basic features")
                result = self._get_basic_features(data)
            
            return result
    
    def _get_categorized_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get features using intelligent categorization."""
        tprint_info("   Using intelligent feature categorization for regime clustering")
        
        # Get regime clustering features
        tprint_info("   Fetching regime clustering feature list...")
        feature_list = get_regime_clustering_features()
        tprint_info(f"   Retrieved {len(feature_list)} feature specifications")
        
        # Generate features using categorizer
        tprint_info("   Generating features with regime categorizer...")
        
        # Get generators for this use case
        generators = self.regime_categorizer.get_generators_for_use_case(FeatureUseCase.REGIME_CLUSTERING)
        tprint_info(f"   Using {len(generators)} generators for regime clustering")
        
        # Generate features from all generators
        all_features = {}
        for generator in generators:
            try:
                gen_features = generator.generate_features(data)
                if gen_features:
                    all_features.update(gen_features)
            except Exception as e:
                tprint_warning(f"   ⚠️ Generator {generator.__class__.__name__} failed: {e}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, index=data.index)
        tprint_info(f"   Generated {features_df.shape[1]} features from categorizer")
        
        # Drop NaN values
        nan_count = features_df.isna().sum().sum()
        if nan_count > 0:
            features_df = features_df.dropna()
            tprint_info(f"   Dropped {nan_count} NaN values, {len(features_df)} samples remain")
        
        # Filter to target range if needed
        tprint_info(f"   Filtering features (target: {self.min_features}-{self.max_features})...")
        
        if features_df.shape[1] > self.max_features:
            # Select features with highest variance
            feature_variance = features_df.var()
            top_features = feature_variance.nlargest(self.max_features).index.tolist()
            features_df = features_df[top_features]
            tprint_info(f"   Reduced from {len(feature_variance)} to {len(top_features)} features (max limit)")
        elif features_df.shape[1] < self.min_features:
            tprint_warning(f"   ⚠️ Only {features_df.shape[1]} features generated (min: {self.min_features})")
        
        tprint_success(f"   ✅ Using {features_df.shape[1]} features for clustering")
        
        return {
            'features': features_df,
            'feature_names': features_df.columns.tolist(),
            'categorized': True,
            'n_features': len(features_df.columns)
        }
    
    def _generate_mtf_regime_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate multi-timeframe (MTF) regime context features.
        
        Provides higher timeframe context (4h, 1d) to improve regime detection on base timeframe (1h).
        
        Args:
            data: Base timeframe market data (e.g., 1h)
            
        Returns:
            DataFrame with MTF regime features or None if generation fails
        """
        try:
            mtf_features = pd.DataFrame(index=data.index)
            
            if 'close' not in data.columns:
                tprint_warning("   No close prices available for MTF features")
                return None
            
            # Ensure data has datetime index for resampling
            if not isinstance(data.index, pd.DatetimeIndex):
                tprint_warning("   Data index is not datetime, skipping MTF features")
                return mtf_features  # Return empty DF
            
            # Resample to higher timeframes and calculate regime indicators
            for tf in self.mtf_timeframes:
                tprint_info(f"   Generating {tf} regime context features...")
                
                # Map timeframe to pandas resample rule
                if tf == '4h':
                    resample_rule = '4H'
                elif tf == '1d' or tf == '24h':
                    resample_rule = '1D'
                elif tf == '1h':
                    resample_rule = '1H'
                elif tf == '2h':
                    resample_rule = '2H'
                elif tf == '8h':
                    resample_rule = '8H'
                elif tf == '12h':
                    resample_rule = '12H'
                else:
                    tprint_warning(f"   Unknown timeframe {tf}, skipping")
                    continue
                
                # Resample OHLCV
                resampled = pd.DataFrame(index=data.index)
                resampled['close'] = data['close'].resample(resample_rule).last().reindex(data.index, method='ffill')
                if 'volume' in data.columns:
                    resampled['volume'] = data['volume'].resample(resample_rule).sum().reindex(data.index, method='ffill')
                
                # Calculate regime indicators on higher timeframe
                returns = resampled['close'].pct_change()
                volatility = returns.rolling(20, min_periods=5).std()
                trend = resampled['close'].pct_change(periods=20)
                
                # Classify into volatility regimes using quantiles (0.33, 0.67)
                vol_q33 = volatility.quantile(0.33)
                vol_q67 = volatility.quantile(0.67)
                vol_regime = pd.Series(
                    np.where(volatility < vol_q33, 0, np.where(volatility < vol_q67, 1, 2)),
                    index=volatility.index
                )
                
                # Classify into trend regimes (-2%, +2% thresholds)
                trend_regime = pd.Series(
                    np.where(trend < -0.02, 0, np.where(trend < 0.02, 1, 2)),  # Down, sideways, up
                    index=trend.index
                )
                
                # Store MTF features
                mtf_features[f'mtf_{tf}_vol_regime'] = vol_regime.fillna(1)  # Default to medium vol
                mtf_features[f'mtf_{tf}_trend_regime'] = trend_regime.fillna(1)  # Default to sideways
                mtf_features[f'mtf_{tf}_volatility'] = volatility.fillna(0)
                mtf_features[f'mtf_{tf}_trend_strength'] = trend.abs().fillna(0)
                
                # Regime alignment (is base timeframe in same regime as higher TF?)
                base_volatility = data['close'].pct_change().rolling(20, min_periods=5).std()
                base_vol_q33 = base_volatility.quantile(0.33)
                base_vol_q67 = base_volatility.quantile(0.67)
                base_vol_regime = pd.Series(
                    np.where(base_volatility < base_vol_q33, 0,
                            np.where(base_volatility < base_vol_q67, 1, 2)),
                    index=data.index
                )
                
                # Alignment indicator (1 if aligned, 0 if not)
                mtf_features[f'mtf_{tf}_vol_aligned'] = (
                    (base_vol_regime.values == mtf_features[f'mtf_{tf}_vol_regime'].values).astype(float)
                )
                
                tprint_info(f"   ✅ {tf}: {5} features (regime, volatility, trend, strength, alignment)")
            
            tprint_success(f"   ✅ Generated {len(mtf_features.columns)} MTF features from {len(self.mtf_timeframes)} timeframes")
            return mtf_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ MTF feature generation failed: {e}")
            self.logger.debug(f"MTF generation error: {e}", exc_info=True) if hasattr(self, 'logger') else None
            return None
    
    def _get_basic_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic features (fallback)."""
        tprint_info("   Generating basic features (fallback mode)")
        
        # Simple returns and volatility
        features = pd.DataFrame(index=data.index)
        
        if 'close' in data.columns:
            tprint_info("   Computing returns and volatility from close prices...")
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['volatility'] = features['returns'].rolling(20).std()
            tprint_info(f"   Added 3 price-based features")
        
        if 'volume' in data.columns:
            tprint_info("   Computing volume features...")
            features['volume_change'] = data['volume'].pct_change()
            tprint_info(f"   Added 1 volume-based feature")
        
        # Drop NaN
        nan_count = features.isna().sum().sum()
        features = features.dropna()
        tprint_info(f"   Dropped {nan_count} NaN values, {len(features)} samples remain")
        
        tprint_success(f"   ✅ Generated {len(features.columns)} basic features")
        
        return {
            'features': features,
            'feature_names': features.columns.tolist(),
            'categorized': False,
            'n_features': len(features.columns)
        }
    
    def cluster_with_sticky_finite_hmm(
        self, 
        data: pd.DataFrame,
        compute_posteriors: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Sticky Finite HMM clustering with feature generation.
        
        Args:
            data: Market data DataFrame
            compute_posteriors: Compute posterior probabilities (False for auto-tuning speedup)
            
        Returns:
            Dictionary with clustering results
        """
        tprint_info("🔍 Starting Sticky Finite HMM clustering with enhanced features")
        
        if not DEPENDENCIES_AVAILABLE:
            tprint_error("❌ Pyro and PyTorch not available")
            raise ImportError("Pyro and PyTorch required. Install: pip install pyro-ppl torch")
        
        # Generate features using comprehensive Feature Bank integration
        tprint_info("🔧 Generating features for clustering")
        feature_result = self.get_comprehensive_clustering_features(data)
        feature_matrix = feature_result['features']
        feature_names = feature_result['feature_names']
        
        tprint_info(f"📊 Feature matrix shape: {feature_matrix.shape}")
        tprint_info(f"📊 Generated {len(feature_names)} features")
        
        # Calculate forward returns from market data for economic validation
        tprint_info("📈 Preparing forward returns for economic validation...")
        forward_returns = None
        if 'close' in data.columns:
            forward_returns = data['close'].pct_change().shift(-1)
            valid_returns = len(forward_returns.dropna())
            tprint_success(f"✅ Calculated forward returns from close prices ({valid_returns} valid values)")
        elif 'returns' in data.columns:
            forward_returns = data['returns'].shift(-1)
            valid_returns = len(forward_returns.dropna())
            tprint_success(f"✅ Using existing returns column ({valid_returns} valid values)")
        else:
            tprint_warning(f"⚠️ No close prices or returns column - economic metrics unavailable")
        
        # Initialize clusterer
        tprint_info("🔧 Initializing Sticky Finite HMM clusterer...")
        config = StickyFiniteHMMConfig(
            K=self.K,
            n_mixtures=self.n_mixtures,
            base_alpha=self.base_alpha,
            kappa=self.kappa,
            num_iters=self.num_iters,
            lr=self.lr,
            enable_pca=self.enable_pca_reduction,
            pca_components=self.pca_components
        )
        tprint_structured({
            "K": config.K,
            "n_mixtures": config.n_mixtures,
            "base_alpha": config.base_alpha,
            "kappa": config.kappa,
            "num_iters": config.num_iters,
            "lr": config.lr,
            "enable_pca": config.enable_pca,
            "pca_components": config.pca_components
        }, level="INFO")
        
        clusterer = StickyFiniteHMMClusterer(config=config)
        tprint_success("✅ Clusterer initialized")
        
        # Run clustering
        with tprint_timer("Sticky Finite HMM Clustering", level="PERFORMANCE"):
            result: StickyFiniteHMMResult = clusterer.fit_predict(
                feature_matrix.values,
                validate=True,
                forward_returns=forward_returns,
                compute_posteriors=compute_posteriors
            )
        
        if not result.success:
            tprint_error(f"❌ Clustering failed: {result.error_message}")
            raise RuntimeError(f"Clustering failed: {result.error_message}")
        
        tprint_success(f"✅ Clustering successful: {result.n_clusters} regimes discovered")
        tprint_info(f"   Final ELBO: {result.final_elbo:.2f}")
        tprint_info(f"   Transition persistence: {result.transition_persistence:.3f}")
        tprint_info(f"   Processing time: {result.processing_time:.2f}s")
        
        # Build return dictionary
        tprint_info("📦 Building results dictionary...")
        return_dict = {
            'cluster_labels': result.cluster_labels,
            'cluster_probabilities': result.cluster_probabilities,
            'n_clusters': result.n_clusters,
            'transition_matrix': result.transition_matrix,
            'emission_params': result.emission_params,
            'cluster_parameters': result.cluster_parameters,
            'state_durations': result.state_durations,
            'final_elbo': result.final_elbo,
            'elbo_history': result.elbo_history,
            'quality_metrics': {
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score,
                'noise_ratio': result.noise_ratio,
                'transition_persistence': result.transition_persistence,
                'composite_score': result.quality_assessment.get('composite_score', 0.0) if result.quality_assessment else 0.0,
                'quality_assessment': result.quality_assessment  # Include full quality assessment for report generation
            },
            'feature_names': feature_names,
            'feature_matrix': feature_matrix,
            'processing_time': result.processing_time,
            'memory_usage_mb': result.memory_usage_mb,
            'metadata': {
                'config': config.__dict__,
                'convergence_info': result.metadata.get('convergence_info', {}) if result.metadata else {},
                'feature_generation': {
                    'n_features_generated': len(feature_names),
                    'feature_categories': False
                }
            },
            'result_object': result
        }
        
        tprint_success(
            f"✅ Sticky Finite HMM clustering complete: {result.n_clusters} regimes, "
            f"ELBO={result.final_elbo:.2f}, "
            f"Quality={return_dict['quality_metrics']['composite_score']:.3f}"
        )
        
        return return_dict


# Convenience function
def perform_enhanced_sticky_finite_hmm_clustering(
    data: pd.DataFrame,
    min_features: int = 50,
    max_features: int = 100,
    K: int = 5,
    base_alpha: float = 0.5,
    kappa: float = 10.0,
    num_iters: int = 800,
    lr: float = 1e-2,
    pca_components: int = 15,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform enhanced Sticky Finite HMM clustering.
    
    Args:
        data: Market data DataFrame
        min_features: Minimum number of features
        max_features: Maximum number of features
        K: Number of states (regimes)
        base_alpha: Concentration for off-diagonal transitions
        kappa: Stickiness parameter
        num_iters: Number of SVI iterations
        lr: Learning rate
        pca_components: Number of PCA components (15-20 recommended)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with clustering results
        
    Note:
        Uses same feature generation pipeline as HDP-HMM for consistency.
    """
    integrator = EnhancedStickyFiniteHMMClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        K=K,
        base_alpha=base_alpha,
        kappa=kappa,
        num_iters=num_iters,
        lr=lr,
        pca_components=pca_components,
        **kwargs
    )
    
    return integrator.cluster_with_sticky_finite_hmm(data)


__all__ = [
    'EnhancedStickyFiniteHMMClusteringIntegration',
    'perform_enhanced_sticky_finite_hmm_clustering'
]

