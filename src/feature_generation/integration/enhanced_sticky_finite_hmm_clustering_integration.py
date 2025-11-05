"""
Enhanced Sticky Finite HMM Clustering Integration

This module provides comprehensive Sticky Finite HMM clustering integration that combines
existing feature bank features with preprocessing for optimal Bayesian regime discovery.

Target: 50-100 comprehensive features optimized for Sticky Finite HMM clustering
Pattern: Mirrors enhanced_hdp_hmm_clustering_integration.py but for fixed-K model
"""

import hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import tprint utilities
from src.utils.tprint import (
    tprint_info, tprint_success, tprint_warning, tprint_error, tprint_data_preview, tprint_timer,
    tprint_performance, tprint_structured, tprint_data_format
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
    tprint_info("✅ Regime-specific features available")
except ImportError as e:
    REGIME_FEATURES_AVAILABLE = False
    tprint_info(
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
    tprint_info("✅ Regime feature categorization available")
except ImportError as e:
    REGIME_CATEGORIZATION_AVAILABLE = False
    tprint_info(f"ℹ️ Regime feature categorization not available: {e}")

# Import regime feature integration (optional)
try:
    from .regime_feature_integration import (
        RegimeFeatureIntegration
    )
    REGIME_INTEGRATION_AVAILABLE = True
    tprint_info("✅ Regime feature integration available")
except ImportError as e:
    REGIME_INTEGRATION_AVAILABLE = False
    tprint_info(f"ℹ️ Regime feature integration not available: {e}")

# Import Sticky Finite HMM components
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
    StickyFiniteHMMClusterer, StickyFiniteHMMConfig, StickyFiniteHMMResult,
    DEPENDENCIES_AVAILABLE
)

# Import ClusterQualityAssessor for comprehensive quality assessment
try:
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
        ClusterQualityAssessor,
        ClusterQualityMetrics
    )
    QUALITY_ASSESSOR_AVAILABLE = True
except ImportError as e:
    QUALITY_ASSESSOR_AVAILABLE = False
    tprint_warning(f"⚠️ ClusterQualityAssessor not available: {e}")


@dataclass
class FeatureCacheConfig:
    """Configuration for feature caching."""
    enable_persistent_cache: bool = False
    enable_instance_cache: bool = True
    cache_key_components: List[str] | None = None
    
    def __post_init__(self):
        if self.cache_key_components is None:
            self.cache_key_components = ['data_hash', 'min_features', 'max_features', 'enable_mtf_features']


class FeatureCacheManager:
    """Dedicated caching utility for feature generation."""
    
    def __init__(self, config: FeatureCacheConfig):
        self.config = config
        self._instance_cache = {}
        self._persistent_cache = {}
        self._data_hash_cache = {}
    
    def generate_cache_key(self, data: pd.DataFrame, **kwargs) -> str:
        """Generate a consistent cache key for feature data.
        
        Args:
            data: Input market data
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            str: Cache key
        """
        # Get or compute data hash
        data_id = id(data)
        if data_id not in self._data_hash_cache:
            if hasattr(data, 'values'):
                data_hash = hashlib.sha256(data.values.tobytes()).hexdigest()
            else:
                data_hash = hashlib.sha256(str(data).encode()).hexdigest()
            self._data_hash_cache[data_id] = data_hash
        else:
            data_hash = self._data_hash_cache[data_id]
        
        # Build cache key from configured components
        key_parts = []
        cache_components = self.config.cache_key_components or []
        for component in cache_components:
            if component == 'data_hash':
                key_parts.append(str(data_hash))
            elif component in kwargs:
                key_parts.append(str(kwargs[component]))
        
        return '_'.join(key_parts) if key_parts else str(data_hash)
    
    def get_cached_result(self, cache_key: str, use_persistent: bool = False):
        """Get cached result if available.
        
        Args:
            cache_key: Cache key to look up
            use_persistent: Whether to check persistent cache
            
        Returns:
            Cached result or None
        """
        if use_persistent and self.config.enable_persistent_cache:
            return self._persistent_cache.get(cache_key)
        elif self.config.enable_instance_cache:
            return self._instance_cache.get(cache_key)
        return None
    
    def cache_result(self, cache_key: str, result: Any, use_persistent: bool = False):
        """Cache a result.
        
        Args:
            cache_key: Cache key for the result
            result: Result to cache
            use_persistent: Whether to store in persistent cache
        """
        if use_persistent and self.config.enable_persistent_cache:
            self._persistent_cache[cache_key] = result
        elif self.config.enable_instance_cache:
            self._instance_cache[cache_key] = result
    
    def clear_cache(self, persistent: bool = False):
        """Clear cache.
        
        Args:
            persistent: Whether to clear persistent cache
        """
        if persistent:
            self._persistent_cache.clear()
        else:
            self._instance_cache.clear()
        self._data_hash_cache.clear()


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
                 min_features: int = 30,  # Reduced from 50 to be more realistic
                 max_features: int = 100,
                 enable_comprehensive_features: bool = True,
                 enable_pca_reduction: bool = True,
                 pca_components: int = 15,  # Use 15 components (same as core model, can use up to 20)
                 K: int = 5,
                 n_mixtures: int = 1,  # Number of Gaussian mixtures per state
                 base_alpha: float = 0.5,
                 kappa: float = 10.0,
                 num_iters: int = 150,  # Reduced from 800 for faster training
                 lr: float = 1e-2,
                 use_regime_categorization: bool = True,
                 use_regime_integration: bool = True,
                 enable_mtf_features: bool = True,  # Enable multi-timeframe regime features
                 mtf_timeframes: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
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
            config: Optional configuration dictionary
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
        
        # Initialize caching system
        cache_config = FeatureCacheConfig(
            enable_persistent_cache=getattr(config, 'persistent_cache_enabled', False),
            enable_instance_cache=True
        )
        self.cache_manager = FeatureCacheManager(cache_config)
        self._persistent_cache_enabled = getattr(config, 'persistent_cache_enabled', False)
        self._persistent_feature_cache = {}  # Legacy support
        self._cached_features = None
        self._cached_data_hash = None

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
        
        # Initialize regime categorizer (lazy loading)
        self._regime_categorizer = None
        self._regime_categorizer_initialized = False
        self._use_regime_categorization = use_regime_categorization
        
        # Initialize regime integration (lazy loading)
        self._regime_integration = None
        self._regime_integration_initialized = False
        self._use_regime_integration = use_regime_integration
    
    @property
    def regime_categorizer(self):
        """Lazy load regime categorizer."""
        if not self._regime_categorizer_initialized and self._use_regime_categorization:
            if REGIME_CATEGORIZATION_AVAILABLE:
                try:
                    self._regime_categorizer = RegimeFeatureCategorizer()
                    tprint_success("✅ Regime feature categorizer initialized (lazy)")
                    self._regime_categorizer_initialized = True
                except Exception as e:
                    tprint_warning(f"⚠️ Could not initialize regime categorizer: {e}")
                    self._regime_categorizer = None
                    self._regime_categorizer_initialized = True
            else:
                self._regime_categorizer = None
                self._regime_categorizer_initialized = True
        return self._regime_categorizer
    
    @property
    def regime_integration(self):
        """Lazy load regime integration."""
        if not self._regime_integration_initialized and self._use_regime_integration:
            if REGIME_INTEGRATION_AVAILABLE:
                try:
                    self._regime_integration = RegimeFeatureIntegration()
                    tprint_success("✅ Regime feature integration initialized (lazy)")
                    self._regime_integration_initialized = True
                except Exception as e:
                    tprint_warning(f"⚠️ Could not initialize regime integration: {e}")
                    self._regime_integration = None
                    self._regime_integration_initialized = True
            else:
                self._regime_integration = None
                self._regime_integration_initialized = True
        return self._regime_integration
    
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

        # Check persistent cache first (across multiple runs)
        # Use fixed feature range for cache key to ensure hits during auto-tuning
        cache_key = self.cache_manager.generate_cache_key(
            data, 
            min_features=50,  # Fixed values for stable cache keys
            max_features=100,
            enable_mtf_features=self.enable_mtf_features
        )

        # Check cache using the cache manager
        cached_result = self.cache_manager.get_cached_result(cache_key, use_persistent=True)
        if cached_result and self._persistent_cache_enabled:
            tprint_info("📋 Using persistent cached features (across runs)")
            return cached_result

        # Check instance cache (within same run)
        cached_result = self.cache_manager.get_cached_result(cache_key, use_persistent=False)
        if cached_result:
            tprint_info("📋 Using cached features (same data)")
            return cached_result
        
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
                        # Ensure MTF features have the same length as original features
                        min_length = min(len(result['features']), len(mtf_features))
                        if len(mtf_features) > len(result['features']):
                            mtf_features = mtf_features.iloc[:min_length]
                        elif len(mtf_features) < len(result['features']):
                            # Pad with NaN if shorter
                            pad_length = len(result['features']) - len(mtf_features)
                            pad_df = pd.DataFrame(np.nan, index=result['features'].index[-pad_length:], columns=mtf_features.columns)
                            mtf_features = pd.concat([mtf_features, pad_df])
                        
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
            
            # Cache the result using the cache manager
            self.cache_manager.cache_result(cache_key, result, use_persistent=self._persistent_cache_enabled)
            if self._persistent_cache_enabled:
                self._persistent_feature_cache[cache_key] = result
                tprint_info("💾 Cached features for future runs (persistent)")
            
            return result
    
    def _get_categorized_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get features using intelligent categorization."""
        tprint_info("   Using intelligent feature categorization for regime clustering")

        # Compute data hash for caching
        cache_key = self.cache_manager.generate_cache_key(data, min_features=50, max_features=100, enable_mtf_features=self.enable_mtf_features)

        # Get regime clustering features
        tprint_info("   Fetching regime clustering feature list...")
        feature_list = get_regime_clustering_features()
        tprint_info(f"   Retrieved {len(feature_list)} feature specifications")
        
        # Generate features using categorizer
        tprint_info("   Generating features with regime categorizer...")
        
        # Get generators for this use case
        generators = self.regime_categorizer.get_generators_for_use_case(FeatureUseCase.REGIME_CLUSTERING)
        tprint_info(f"   Using {len(generators)} generators for regime clustering")
        
        # Use enhanced vectorized feature generation with multiple optimization backends
        tprint_info("   🚀 Using enhanced vectorized feature generation with multiple backends")
        
        # Initialize optimization managers
        vectorization_manager = None
        rolling_optimizer = None
        statistical_optimizer = None
        
        try:
            # Try to import and initialize UnifiedVectorizationManager
            from src.utils.ml_common.unified_vectorization_manager import (
                UnifiedVectorizationManager, OperationType, OperationConfig
            )
            vectorization_manager = UnifiedVectorizationManager()
            tprint_info("   ✅ UnifiedVectorizationManager initialized")
        except ImportError as e:
            tprint_warning(f"   ⚠️ UnifiedVectorizationManager not available: {e}")
            OperationType = None
            OperationConfig = None
        
        try:
            # Try to import and initialize VectorBTRollingOptimizer
            from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
            rolling_optimizer = get_vectorbt_rolling_optimizer()
            if rolling_optimizer:
                tprint_info("   ✅ VectorBTRollingOptimizer initialized")
        except ImportError as e:
            tprint_warning(f"   ⚠️ VectorBTRollingOptimizer not available: {e}")
        
        try:
            # Try to import and initialize StatisticalCalculationsOptimizer
            from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor
            statistical_optimizer = BatchMatrixProcessor(enable_gpu=True, enable_parallel=True)
            tprint_info("   ✅ StatisticalCalculationsOptimizer initialized")
        except ImportError as e:
            tprint_warning(f"   ⚠️ StatisticalCalculationsOptimizer not available: {e}")
        
        # Try optimized pipeline first
        features_df = None
        
        if vectorization_manager is not None and OperationType is not None and OperationConfig is not None:
            try:
                # Use UnifiedVectorizationManager for feature engineering
                operation_config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape,
                    memory_budget_mb=1024.0,
                    time_budget_seconds=300.0
                )
                
                # Generate features using the unified manager
                tprint_info("   🔄 Generating features with UnifiedVectorizationManager...")
                features_df = self._generate_features_with_vectorization_manager(
                    data, vectorization_manager, operation_config, OperationType.FEATURE_ENGINEERING
                )
                tprint_success(f"   ✅ Generated {len(features_df.columns)} features using UnifiedVectorizationManager")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ UnifiedVectorizationManager failed: {e}")
                features_df = None
        
        # Fallback to VectorBT rolling optimizer if available
        if features_df is None and rolling_optimizer is not None:
            try:
                tprint_info("   🔄 Generating features with VectorBTRollingOptimizer...")
                features_df = self._generate_features_with_vectorbt(data, rolling_optimizer)
                tprint_success(f"   ✅ Generated {len(features_df.columns)} features using VectorBTRollingOptimizer")
                
                # Check if we have enough features, if not, enhance with additional features
                if len(features_df.columns) < self.min_features:
                    tprint_info(f"   ℹ️ VectorBT generated {len(features_df.columns)} features (< {self.min_features}), enhancing with additional features")
                    additional_features = self._generate_additional_vectorbt_features(data, rolling_optimizer)
                    if additional_features is not None and len(additional_features.columns) > 0:
                        # Combine with existing features
                        features_df = pd.concat([features_df, additional_features], axis=1)
                        tprint_success(f"   ✅ Enhanced to {len(features_df.columns)} total features")
                    
                    # If still not enough features, fall back to comprehensive generators
                    if len(features_df.columns) < self.min_features:
                        tprint_info(f"   ℹ️ Still only {len(features_df.columns)} features (< {self.min_features}), falling back to comprehensive generators")
                        features_df = None  # Force fallback to get more features
                
            except Exception as e:
                tprint_warning(f"   ⚠️ VectorBTRollingOptimizer failed: {e}")
                features_df = None
        
        # Fallback to statistical optimizer if available
        if features_df is None and statistical_optimizer is not None:
            try:
                tprint_info("   🔄 Generating features with StatisticalCalculationsOptimizer...")
                features_df = self._generate_features_with_statistical_optimizer(data, statistical_optimizer)
                tprint_success(f"   ✅ Generated {len(features_df.columns)} features using StatisticalCalculationsOptimizer")
                
                # Check if we have enough features, if not, use the comprehensive regime generators
                if len(features_df.columns) < self.min_features:
                    tprint_warning(f"   ⚠️ Statistical optimizer only generated {len(features_df.columns)} features (< {self.min_features}), using comprehensive regime generators")
                    # Use the existing regime generators instead of manual implementation
                    features_df = self._generate_features_with_regime_generators(data, generators)
                    tprint_success(f"   ✅ Generated {len(features_df.columns)} features using comprehensive regime generators")
                else:
                    tprint_success(f"   ✅ Statistical optimizer generated {len(features_df.columns)} features (meets minimum {self.min_features})")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ StatisticalCalculationsOptimizer failed: {e}")
                features_df = None
        
        # Final fallback to comprehensive regime generators
        if features_df is None:
            tprint_info("   ℹ️ All optimized methods failed, using comprehensive regime generators as final fallback")
            features_df = self._generate_features_with_regime_generators(data, generators)
        
        # Handle NaN values more intelligently
        tprint_info("   Analyzing NaN values in generated features...")
        
        # Calculate NaN percentages for each feature
        nan_percentages = (features_df.isna().sum() / len(features_df) * 100).sort_values(ascending=False)
        high_nan_features = nan_percentages[nan_percentages > 50.0]
        
        if len(high_nan_features) > 0:
            tprint_warning(f"   ⚠️ Found {len(high_nan_features)} features with >50% NaN values:")
            for feature, nan_pct in high_nan_features.head(10).items():
                tprint_info(f"      • {feature}: {nan_pct:.1f}% NaN")
            
            # Option 1: Drop features with extremely high NaN percentages
            features_df = features_df.drop(columns=high_nan_features.index)
            tprint_info(f"   Dropped {len(high_nan_features)} high-NaN features")
        
        # Drop remaining NaN values (should be much fewer now)
        remaining_nan_count = features_df.isna().sum().sum()
        if remaining_nan_count > 0:
            initial_samples = len(features_df)
            features_df = features_df.dropna()
            dropped_samples = initial_samples - len(features_df)
            tprint_info(f"   Dropped {remaining_nan_count} remaining NaN values, {dropped_samples} samples lost")
        
        tprint_success(f"   ✅ Final dataset: {len(features_df)} samples, {features_df.shape[1]} features")
        
        # Filter to target range if needed
        tprint_info(f"   Filtering features (target: {self.min_features}-{self.max_features})...")
        
        if features_df.shape[1] > self.max_features:
            # Select features with highest variance
            feature_variance = features_df.var()
            # Ensure feature_variance is a Series (not scalar)
            if isinstance(feature_variance, pd.Series):
                top_features = feature_variance.nlargest(self.max_features).index.tolist()
            else:
                # Fallback: just take first max_features columns
                top_features = features_df.columns[:self.max_features].tolist()
            features_df = features_df[top_features]
            tprint_info(f"   Reduced from {len(feature_variance) if hasattr(feature_variance, '__len__') else features_df.shape[1]} to {len(top_features)} features (max limit)")
        elif features_df.shape[1] < self.min_features:
            tprint_warning(f"   ⚠️ Only {features_df.shape[1]} features generated (min: {self.min_features})")
        
        tprint_success(f"   ✅ Using {features_df.shape[1]} features for clustering")
        
        # Cache the result
        result = {
            'features': features_df,
            'feature_names': features_df.columns.tolist(),
            'categorized': True,
            'n_features': len(features_df.columns)
        }

        self._cached_features = result
        self._cached_data_hash = cache_key
        tprint_info("💾 Cached features for future trials")
        
        # Cache the result in both instance and persistent caches
        self.cache_manager.cache_result(cache_key, result, use_persistent=self._persistent_cache_enabled)
        if self._persistent_cache_enabled:
            self._persistent_feature_cache[cache_key] = result
            tprint_info("💾 Cached features for future runs (persistent)")

        return result
    
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
            # Ensure data has datetime index for resampling
            if not isinstance(data.index, pd.DatetimeIndex):
                # Check if timestamp column exists and convert it to datetime index
                if 'timestamp' in data.columns:
                    tprint_info("   Converting timestamp column to datetime index for MTF features")
                    data = data.set_index(pd.to_datetime(data['timestamp']))
                    # Drop the original timestamp column to avoid duplication
                    data = data.drop('timestamp', axis=1)
                else:
                    # Create synthetic datetime index for MTF features
                    tprint_info("   Creating synthetic datetime index for MTF features (15-minute intervals)")
                    # Assume 15-minute intervals for crypto data
                    start_time = pd.Timestamp.now().floor('15T') - pd.Timedelta(minutes=len(data) * 15)
                    data.index = pd.date_range(start=start_time, periods=len(data), freq='15T')
                    tprint_success(f"   ✅ Created datetime index: {data.index[0]} to {data.index[-1]}")
            
            mtf_features = pd.DataFrame(index=data.index)
            
            if 'close' not in data.columns:
                tprint_warning("   No close prices available for MTF features")
                return mtf_features
            
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
                
                # Fill NaN values early to reduce propagation
                volatility = volatility.fillna(method='bfill').fillna(0)
                trend = trend.fillna(method='bfill').fillna(0)
                
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
                
                # Store MTF features with better NaN handling
                mtf_features[f'mtf_{tf}_vol_regime'] = vol_regime.fillna(1)  # Default to medium vol
                mtf_features[f'mtf_{tf}_trend_regime'] = trend_regime.fillna(1)  # Default to sideways
                mtf_features[f'mtf_{tf}_volatility'] = volatility.fillna(0)
                mtf_features[f'mtf_{tf}_trend_strength'] = trend.abs().fillna(0)
                
                # Regime alignment (is base timeframe in same regime as higher TF?)
                base_volatility = data['close'].pct_change().rolling(20, min_periods=5).std()
                base_volatility = base_volatility.fillna(method='bfill').fillna(0)
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
    
    def _generate_features_with_vectorization_manager(self, data: pd.DataFrame, vectorization_manager, operation_config: Dict, operation_type) -> pd.DataFrame:
        """Generate features using UnifiedVectorizationManager."""
        try:
            # Use the vectorization manager to optimize feature generation
            # Create a simple feature generation function that the manager can optimize
            def generate_basic_features(df):
                """Basic feature generation function for optimization."""
                features = {}
                
                # Price-based features
                if 'close' in df.columns:
                    returns = df['close'].pct_change()
                    features['returns'] = returns
                    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                    features['volatility_20'] = returns.rolling(20).std()
                    features['volatility_50'] = returns.rolling(50).std()
                    
                    # Trend features
                    features['sma_20'] = df['close'].rolling(20).mean()
                    features['sma_50'] = df['close'].rolling(50).mean()
                    features['price_sma_ratio_20'] = df['close'] / features['sma_20']
                    features['price_sma_ratio_50'] = df['close'] / features['sma_50']
                
                # Volume features
                if 'volume' in df.columns:
                    features['volume_sma_20'] = df['volume'].rolling(20).mean()
                    features['volume_ratio'] = df['volume'] / features['volume_sma_20']
                    features['volume_change'] = df['volume'].pct_change()
                
                # High-Low features
                if all(col in df.columns for col in ['high', 'low']):
                    features['hl_ratio'] = (df['high'] - df['low']) / df['close']
                    features['hl_ratio_20'] = features['hl_ratio'].rolling(20).mean()
                
                # Open-Close features
                if all(col in df.columns for col in ['open', 'close']):
                    features['oc_ratio'] = (df['close'] - df['open']) / df['open']
                    features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                
                return pd.DataFrame(features)
            
            # Execute with optimization - pass data through the operation function
            result = vectorization_manager.optimize_operation(
                operation_type=operation_type,
                data=data,
                config=operation_config
            )
            
            features_df = result.result if hasattr(result, 'result') else generate_basic_features(data)
            
            # Check if we have enough features, if not, return None to trigger fallback
            if features_df is not None and len(features_df.columns) < self.min_features:
                tprint_info(f"   ℹ️ UnifiedVectorizationManager generated only {len(features_df.columns)} features (< {self.min_features}), triggering fallback generators")
                return None
            
            return features_df
            
        except Exception as e:
            raise Exception(f"Vectorization manager failed: {e}")
    
    def _generate_features_with_vectorbt(self, data: pd.DataFrame, rolling_optimizer) -> pd.DataFrame:
        """Generate features using VectorBTRollingOptimizer."""
        try:
            # Use VectorBT for optimized rolling calculations
            features = {}
            
            # Price-based features with VectorBT optimization
            if 'close' in data.columns:
                # Returns
                features['returns'] = data['close'].pct_change()
                features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                
                # Use VectorBT for optimized rolling operations
                if hasattr(rolling_optimizer, 'optimized_rolling_std'):
                    features['volatility_20'] = rolling_optimizer.optimized_rolling_std(features['returns'], window=20)
                    features['volatility_50'] = rolling_optimizer.optimized_rolling_std(features['returns'], window=50)
                else:
                    # Fallback to pandas
                    features['volatility_20'] = features['returns'].rolling(20).std()
                    features['volatility_50'] = features['returns'].rolling(50).std()
                
                # Moving averages
                if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                    features['sma_20'] = rolling_optimizer.optimized_rolling_mean(data['close'], window=20)
                    features['sma_50'] = rolling_optimizer.optimized_rolling_mean(data['close'], window=50)
                else:
                    features['sma_20'] = data['close'].rolling(20).mean()
                    features['sma_50'] = data['close'].rolling(50).mean()
                
                features['price_sma_ratio_20'] = data['close'] / features['sma_20']
                features['price_sma_ratio_50'] = data['close'] / features['sma_50']
                
                # Additional price features
                features['price_momentum_5'] = data['close'] / data['close'].shift(5)
                features['price_momentum_10'] = data['close'] / data['close'].shift(10)
                features['price_momentum_20'] = data['close'] / data['close'].shift(20)
                
                # Price position relative to moving averages
                features['price_above_sma_20'] = (data['close'] > features['sma_20']).astype(int)
                features['price_above_sma_50'] = (data['close'] > features['sma_50']).astype(int)
                
                # Moving average crossovers
                features['sma_crossover'] = (features['sma_20'] > features['sma_50']).astype(int)
                
                # Volatility features
                features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']
                features['volatility_change'] = features['volatility_20'].pct_change()
                
                # High-Low features
                if 'high' in data.columns and 'low' in data.columns:
                    features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
                    features['high_close_ratio'] = (data['high'] - data['close']) / data['close']
                    features['low_close_ratio'] = (data['close'] - data['low']) / data['close']
            
            # Volume features
            if 'volume' in data.columns:
                if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                    features['volume_sma_10'] = rolling_optimizer.optimized_rolling_mean(data['volume'], window=10)
                    features['volume_sma_20'] = rolling_optimizer.optimized_rolling_mean(data['volume'], window=20)
                    features['volume_sma_50'] = rolling_optimizer.optimized_rolling_mean(data['volume'], window=50)
                else:
                    features['volume_sma_10'] = data['volume'].rolling(10).mean()
                    features['volume_sma_20'] = data['volume'].rolling(20).mean()
                    features['volume_sma_50'] = data['volume'].rolling(50).mean()
                
                features['volume_ratio_10'] = data['volume'] / features['volume_sma_10']
                features['volume_ratio_20'] = data['volume'] / features['volume_sma_20']
                features['volume_ratio_50'] = data['volume'] / features['volume_sma_50']
                features['volume_change'] = data['volume'].pct_change()
                features['volume_momentum_5'] = data['volume'] / data['volume'].shift(5)
                features['volume_momentum_10'] = data['volume'] / data['volume'].shift(10)
                
                # Volume volatility
                features['volume_volatility_10'] = features['volume_change'].rolling(10).std()
                features['volume_volatility_20'] = features['volume_change'].rolling(20).std()
            
            # OHLC features if available
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                features['open_close_ratio'] = data['open'] / data['close']
                features['high_open_ratio'] = data['high'] / data['open']
                features['low_open_ratio'] = data['low'] / data['open']
                features['body_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'])
                features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
                features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
            
            return pd.DataFrame(features)
            
        except Exception as e:
            raise Exception(f"VectorBT optimizer failed: {e}")
    
    def _generate_additional_vectorbt_features(self, data: pd.DataFrame, rolling_optimizer) -> Optional[pd.DataFrame]:
        """
        Generate additional features using VectorBT when the initial set is insufficient.
        
        This method creates more diverse features to reach the minimum threshold.
        
        Args:
            data: Market data DataFrame
            rolling_optimizer: VectorBT rolling optimizer instance
            
        Returns:
            DataFrame with additional features or None if generation fails
        """
        try:
            additional_features = {}
            
            # Price-based additional features
            if 'close' in data.columns:
                close_values = data['close'].values
                
                # Additional momentum features with different periods
                for period in [3, 7, 14, 21, 30, 60]:
                    if len(data) >= period:
                        momentum = close_values / np.pad(close_values[:-period], (period, 0), constant_values=close_values[0])
                        additional_features[f'momentum_{period}'] = momentum
                        
                        # Rate of change
                        roc = ((close_values - np.pad(close_values[:-period], (period, 0), constant_values=close_values[0])) /
                               (np.pad(close_values[:-period], (period, 0), constant_values=close_values[0]) + 1e-10)) * 100
                        additional_features[f'roc_{period}'] = roc
                
                # Additional volatility features with different periods
                returns = np.diff(close_values) / close_values[:-1]
                returns = np.pad(returns, (1, 0), constant_values=np.nan)
                
                for period in [5, 10, 15, 25, 30, 40]:
                    if len(returns) >= period:
                        if hasattr(rolling_optimizer, 'optimized_rolling_std'):
                            vol = rolling_optimizer.optimized_rolling_std(returns, window=period)
                        else:
                            vol = returns.rolling(period).std()
                        additional_features[f'volatility_{period}'] = vol
                
                # Additional moving averages and crossovers
                for period in [8, 13, 21, 34, 55, 89]:  # Fibonacci periods
                    if len(data) >= period:
                        if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                            sma = rolling_optimizer.optimized_rolling_mean(data['close'], window=period)
                        else:
                            sma = data['close'].rolling(period).mean()
                        additional_features[f'sma_{period}'] = sma
                        
                        # Price to SMA ratio
                        additional_features[f'price_sma_{period}_ratio'] = data['close'] / (sma + 1e-10)
                
                # EMA features with different periods
                for period in [5, 8, 12, 21, 26, 50]:
                    if len(data) >= period:
                        ema = self._calculate_ema(close_values, period)
                        additional_features[f'ema_{period}'] = ema
                        additional_features[f'price_ema_{period}_ratio'] = close_values / (ema + 1e-10)
                
                # Bollinger Bands with different periods
                for period in [10, 20, 30]:
                    if len(data) >= period:
                        if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                            sma = rolling_optimizer.optimized_rolling_mean(data['close'], window=period)
                            if hasattr(rolling_optimizer, 'optimized_rolling_std'):
                                std = rolling_optimizer.optimized_rolling_std(data['close'].pct_change(), window=period)
                            else:
                                std = data['close'].pct_change().rolling(period).std()
                        else:
                            sma = data['close'].rolling(period).mean()
                            std = data['close'].rolling(period).std()
                        
                        additional_features[f'bb_upper_{period}'] = sma + 2 * std
                        additional_features[f'bb_lower_{period}'] = sma - 2 * std
                        additional_features[f'bb_width_{period}'] = (sma + 2 * std - (sma - 2 * std)) / (sma + 1e-10)
                        additional_features[f'bb_position_{period}'] = (data['close'] - (sma - 2 * std)) / (4 * std + 1e-10)
                
                # RSI with different periods
                for period in [7, 14, 21]:
                    if len(returns) >= period:
                        gains = np.where(returns > 0, returns, 0)
                        losses = np.where(returns < 0, -returns, 0)
                        
                        if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                            avg_gain = rolling_optimizer.optimized_rolling_mean(gains, window=period)
                            avg_loss = rolling_optimizer.optimized_rolling_mean(losses, window=period)
                        else:
                            avg_gain = gains.rolling(period).mean()
                            avg_loss = losses.rolling(period).mean()
                        
                        rs = avg_gain / (avg_loss + 1e-10)
                        rsi = 100 - (100 / (1 + rs))
                        additional_features[f'rsi_{period}'] = rsi
                
                # Stochastic Oscillator with different periods
                if all(col in data.columns for col in ['high', 'low']):
                    for period in [5, 10, 14, 20]:
                        if len(data) >= period:
                            if hasattr(rolling_optimizer, 'optimized_rolling_max'):
                                highest = rolling_optimizer.optimized_rolling_max(data['high'], window=period)
                                lowest = rolling_optimizer.optimized_rolling_min(data['low'], window=period)
                            else:
                                highest = data['high'].rolling(period).max()
                                lowest = data['low'].rolling(period).min()
                            
                            additional_features[f'stoch_k_{period}'] = ((data['close'] - lowest) / (highest - lowest + 1e-10)) * 100
                            
                            # Stochastic D (smoothed K)
                            if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                                stoch_d = rolling_optimizer.optimized_rolling_mean(additional_features[f'stoch_k_{period}'], window=3)
                            else:
                                stoch_d = additional_features[f'stoch_k_{period}'].rolling(3).mean()
                            additional_features[f'stoch_d_{period}'] = stoch_d
            
            # Volume-based additional features
            if 'volume' in data.columns:
                volume_values = data['volume'].values
                
                # Volume-weighted price features
                if 'close' in data.columns:
                    vwap_5 = self._calculate_vwap(close_values, volume_values, 5)
                    vwap_10 = self._calculate_vwap(close_values, volume_values, 10)
                    vwap_20 = self._calculate_vwap(close_values, volume_values, 20)
                    
                    additional_features[f'vwap_5'] = vwap_5
                    additional_features[f'vwap_10'] = vwap_10
                    additional_features[f'vwap_20'] = vwap_20
                    
                    additional_features[f'price_vwap_5_ratio'] = close_values / (vwap_5 + 1e-10)
                    additional_features[f'price_vwap_10_ratio'] = close_values / (vwap_10 + 1e-10)
                    additional_features[f'price_vwap_20_ratio'] = close_values / (vwap_20 + 1e-10)
                
                # Volume profile features
                for period in [5, 10, 20, 30]:
                    if len(data) >= period:
                        if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                            vol_sma = rolling_optimizer.optimized_rolling_mean(data['volume'], window=period)
                            vol_std = rolling_optimizer.optimized_rolling_std(data['volume'].pct_change(), window=period)
                        else:
                            vol_sma = data['volume'].rolling(period).mean()
                            vol_std = data['volume'].rolling(period).std()
                        
                        additional_features[f'volume_sma_{period}'] = vol_sma
                        additional_features[f'volume_std_{period}'] = vol_std
                        additional_features[f'volume_ratio_{period}'] = data['volume'] / (vol_sma + 1e-10)
                        additional_features[f'volume_zscore_{period}'] = (data['volume'] - vol_sma) / (vol_std + 1e-10)
            
            # Price-volume interaction features
            if 'close' in data.columns and 'volume' in data.columns:
                for period in [5, 10, 20]:
                    if len(data) >= period:
                        # Price-Volume correlation
                        if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                            price_vol_corr = self._calculate_rolling_corr(close_values, volume_values, period)
                        else:
                            price_vol_corr = data['close'].rolling(period).corr(data['volume'])
                        
                        additional_features[f'price_volume_corr_{period}'] = price_vol_corr
                        
                        # Volume-weighted returns
                        weighted_returns = returns * volume_values
                        if hasattr(rolling_optimizer, 'optimized_rolling_mean'):
                            vol_weighted_returns = rolling_optimizer.optimized_rolling_mean(weighted_returns, window=period)
                        else:
                            vol_weighted_returns = weighted_returns.rolling(period).mean()
                        
                        additional_features[f'vol_weighted_returns_{period}'] = vol_weighted_returns
            
            # Convert to DataFrame
            if additional_features:
                additional_df = pd.DataFrame(additional_features, index=data.index)
                
                # Remove any constant or NaN features
                additional_df = additional_df.loc[:, additional_df.nunique() > 1]  # Remove constant features
                additional_df = additional_df.dropna(axis=1, how='all')  # Remove all-NaN columns
                
                tprint_success(f"   ✅ Generated {len(additional_df.columns)} additional VectorBT features")
                return additional_df
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"   ❌ Additional VectorBT feature generation failed: {e}")
            return None
    
    def _generate_features_with_statistical_optimizer(self, data: pd.DataFrame, statistical_optimizer) -> pd.DataFrame:
        """Generate features using StatisticalCalculationsOptimizer."""
        try:
            tprint_info("   📊 Using statistical optimizer for feature generation...")
            features = {}

            # Get data length to ensure all features have the same length
            data_length = len(data)
            tprint_info(f"      📏 Data length: {data_length}")

            # Convert to numpy arrays once for efficiency
            close_values = data['close'].values if 'close' in data.columns else None
            if close_values is not None:
                tprint_info(f"      📈 Close values length: {len(close_values)}")

            # Cache rolling calculations to avoid repetition
            rolling_cache = {}

            # Price-based features
            if close_values is not None:
                try:
                    returns = np.diff(close_values) / close_values[:-1]
                    returns = np.pad(returns, (1, 0), constant_values=np.nan)
                    # Ensure returns has the correct length
                    if len(returns) != data_length:
                        tprint_warning(f"      ⚠️ Returns length mismatch: {len(returns)} vs {data_length}")
                        returns = returns[:data_length] if len(returns) > data_length else np.pad(returns, (0, data_length - len(returns)), constant_values=np.nan)
                        tprint_info(f"      ✅ Returns length corrected to: {len(returns)}")
                    features['returns'] = returns

                    log_returns = np.log(close_values / np.pad(close_values[:-1], (1, 0), constant_values=close_values[0]))
                    if len(log_returns) != data_length:
                        tprint_warning(f"      ⚠️ Log returns length mismatch: {len(log_returns)} vs {data_length}")
                        log_returns = log_returns[:data_length] if len(log_returns) > data_length else np.pad(log_returns, (0, data_length - len(log_returns)), constant_values=np.nan)
                        tprint_info(f"      ✅ Log returns length corrected to: {len(log_returns)}")
                    features['log_returns'] = log_returns

                    # Cache rolling calculations
                    close_array = np.asarray(close_values)

                    # Use statistical optimizer if available, otherwise numpy with caching
                    if hasattr(statistical_optimizer, 'get_rolling_stats') and len(data) >= 50:
                        tprint_info("      🔄 Using statistical optimizer rolling stats...")
                        try:
                            rolling_stats = statistical_optimizer.get_rolling_stats(close_array)
                            tprint_info(f"      📊 Rolling stats keys: {list(rolling_stats.keys())}")

                            for key, value in rolling_stats.items():
                                tprint_info(f"         {key}: shape={value.shape}")

                            features['volatility_20'] = np.pad(rolling_stats['std_20'], (19, 0), constant_values=np.nan)[:data_length]
                            features['volatility_50'] = np.pad(rolling_stats['std_50'], (49, 0), constant_values=np.nan)[:data_length]

                            # Cache the rolling means
                            rolling_cache['sma_20'] = np.pad(rolling_stats['mean_20'], (19, 0), constant_values=np.nan)[:data_length]
                            rolling_cache['sma_50'] = np.pad(rolling_stats['mean_50'], (49, 0), constant_values=np.nan)[:data_length]
                            tprint_info("      ✅ Statistical optimizer rolling stats processed successfully")
                        except Exception as stats_error:
                            tprint_warning(f"      ❌ Statistical optimizer failed: {stats_error}")
                            tprint_info("      🔄 Falling back to numpy rolling calculations...")
                            # Fallback to numpy-based rolling with caching
                            rolling_cache['sma_20'] = self._numpy_rolling_mean(close_array, 20)
                            rolling_cache['sma_50'] = self._numpy_rolling_mean(close_array, 50)

                            features['volatility_20'] = self._numpy_rolling_std(returns, 20)
                            features['volatility_50'] = self._numpy_rolling_std(returns, 50)
                    else:
                        tprint_info("      🔄 Using numpy rolling calculations (fallback)...")
                        # Fallback to numpy-based rolling with caching
                        rolling_cache['sma_20'] = self._numpy_rolling_mean(close_array, 20)
                        rolling_cache['sma_50'] = self._numpy_rolling_mean(close_array, 50)

                        features['volatility_20'] = self._numpy_rolling_std(returns, 20)
                        features['volatility_50'] = self._numpy_rolling_std(returns, 50)

                except Exception as price_error:
                    tprint_warning(f"      ❌ Price-based features failed: {price_error}")
                    import traceback
                    tprint_warning(f"         🔍 Traceback: {traceback.format_exc()}")
                    raise Exception(f"Statistical optimizer failed: {price_error}")

                # Use cached rolling means
                features['sma_20'] = rolling_cache['sma_20']
                features['sma_50'] = rolling_cache['sma_50']
                features['price_sma_ratio_20'] = close_values / rolling_cache['sma_20']
                features['price_sma_ratio_50'] = close_values / rolling_cache['sma_50']

                # Add more features to reach target count
                # Momentum features
                momentum_5 = close_values / np.pad(close_values[:-5], (5, 0), constant_values=close_values[0])
                features['momentum_5'] = momentum_5[:data_length] if len(momentum_5) > data_length else np.pad(momentum_5, (0, data_length - len(momentum_5)), constant_values=np.nan)

                momentum_10 = close_values / np.pad(close_values[:-10], (10, 0), constant_values=close_values[0])
                features['momentum_10'] = momentum_10[:data_length] if len(momentum_10) > data_length else np.pad(momentum_10, (0, data_length - len(momentum_10)), constant_values=np.nan)

                # RSI-like features
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                avg_gain_14 = self._numpy_rolling_mean(gains, 14)
                avg_loss_14 = self._numpy_rolling_mean(losses, 14)
                rs = avg_gain_14 / (avg_loss_14 + 1e-10)
                features['rsi_14'] = 100 - (100 / (1 + rs))

                # MACD-like features
                ema_12 = self._calculate_ema(np.asarray(close_values), 12)
                ema_26 = self._calculate_ema(np.asarray(close_values), 26)
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = self._calculate_ema(np.asarray(features['macd']), 9)

                # Bollinger Bands
                sma_20 = rolling_cache['sma_20']
                std_20 = features['volatility_20']
                features['bb_upper'] = sma_20 + 2 * std_20
                features['bb_lower'] = sma_20 - 2 * std_20
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20

                # Additional volatility features
                features['volatility_10'] = self._numpy_rolling_std(returns, 10)
                features['volatility_30'] = self._numpy_rolling_std(returns, 30)

                # Trend strength indicators
                trend_20 = (close_values - np.pad(close_values[:-20], (20, 0), constant_values=close_values[0])) / close_values
                features['trend_strength_20'] = trend_20[:data_length] if len(trend_20) > data_length else np.pad(trend_20, (0, data_length - len(trend_20)), constant_values=np.nan)

                trend_50 = (close_values - np.pad(close_values[:-50], (50, 0), constant_values=close_values[0])) / close_values
                features['trend_strength_50'] = trend_50[:data_length] if len(trend_50) > data_length else np.pad(trend_50, (0, data_length - len(trend_50)), constant_values=np.nan)

            # Volume features with caching
            if 'volume' in data.columns:
                volume_values = data['volume'].values
                volume_array = np.asarray(volume_values)

                # Cache volume SMA
                rolling_cache['volume_sma_20'] = self._numpy_rolling_mean(volume_array, 20)

                features['volume_sma_20'] = rolling_cache['volume_sma_20']
                features['volume_ratio'] = volume_values / rolling_cache['volume_sma_20']
                volume_change = np.diff(volume_values) / volume_values[:-1]
                volume_change_padded = np.pad(volume_change, (1, 0), constant_values=np.nan)
                features['volume_change'] = volume_change_padded[:data_length] if len(volume_change_padded) > data_length else np.pad(volume_change_padded, (0, data_length - len(volume_change_padded)), constant_values=np.nan)

                # Additional volume features
                features['volume_sma_10'] = self._numpy_rolling_mean(volume_array, 10)
                features['volume_sma_50'] = self._numpy_rolling_mean(volume_array, 50)
                features['volume_volatility_20'] = self._numpy_rolling_std(volume_change_padded, 20)

                volume_trend_20 = volume_values / np.pad(volume_values[:-20], (20, 0), constant_values=volume_values[0])
                features['volume_trend_20'] = volume_trend_20[:data_length] if len(volume_trend_20) > data_length else np.pad(volume_trend_20, (0, data_length - len(volume_trend_20)), constant_values=np.nan)

                # Volume-price interaction features
                features['volume_price_corr_20'] = self._calculate_rolling_corr(np.asarray(volume_values), np.asarray(close_values), 20)
                features['volume_price_corr_50'] = self._calculate_rolling_corr(np.asarray(volume_values), np.asarray(close_values), 50)

                # Volume-based momentum
                vol_momentum_5 = volume_values / np.pad(volume_values[:-5], (5, 0), constant_values=volume_values[0])
                features['volume_momentum_5'] = vol_momentum_5[:data_length] if len(vol_momentum_5) > data_length else np.pad(vol_momentum_5, (0, data_length - len(vol_momentum_5)), constant_values=np.nan)

                vol_momentum_10 = volume_values / np.pad(volume_values[:-10], (10, 0), constant_values=volume_values[0])
                features['volume_momentum_10'] = vol_momentum_10[:data_length] if len(vol_momentum_10) > data_length else np.pad(vol_momentum_10, (0, data_length - len(vol_momentum_10)), constant_values=np.nan)

            # Additional price features to reach minimum count
            if close_values is not None:
                # Additional momentum indicators
                momentum_20 = close_values / np.pad(close_values[:-20], (20, 0), constant_values=close_values[0])
                features['momentum_20'] = momentum_20[:data_length] if len(momentum_20) > data_length else np.pad(momentum_20, (0, data_length - len(momentum_20)), constant_values=np.nan)

                # Price rate of change
                roc_5 = ((close_values - np.pad(close_values[:-5], (5, 0), constant_values=close_values[0])) / np.pad(close_values[:-5], (5, 0), constant_values=close_values[0])) * 100
                features['roc_5'] = roc_5[:data_length] if len(roc_5) > data_length else np.pad(roc_5, (0, data_length - len(roc_5)), constant_values=np.nan)

                roc_10 = ((close_values - np.pad(close_values[:-10], (10, 0), constant_values=close_values[0])) / np.pad(close_values[:-10], (10, 0), constant_values=close_values[0])) * 100
                features['roc_10'] = roc_10[:data_length] if len(roc_10) > data_length else np.pad(roc_10, (0, data_length - len(roc_10)), constant_values=np.nan)

                # Williams %R
                highest_14 = self._numpy_rolling_max(close_values, 14)
                lowest_14 = self._numpy_rolling_min(close_values, 14)
                features['williams_r_14'] = ((highest_14 - close_values) / (highest_14 - lowest_14 + 1e-10)) * -100

                # Stochastic Oscillator
                features['stoch_k_14'] = ((close_values - lowest_14) / (highest_14 - lowest_14 + 1e-10)) * 100
                features['stoch_d_14'] = self._numpy_rolling_mean(features['stoch_k_14'], 3)

                # Average True Range (ATR)
                if all(col in data.columns for col in ['high', 'low']):
                    high_values = data['high'].values
                    low_values = data['low'].values
                    
                    tr1 = high_values - low_values
                    tr2 = np.abs(high_values - np.pad(close_values[:-1], (1, 0), constant_values=close_values[0]))
                    tr3 = np.abs(low_values - np.pad(close_values[:-1], (1, 0), constant_values=close_values[0]))
                    
                    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
                    features['atr_14'] = self._numpy_rolling_mean(true_range, 14)

                # Commodity Channel Index (CCI)
                typical_price = (close_values + high_values + low_values) / 3 if all(col in data.columns for col in ['high', 'low']) else close_values
                sma_tp_20 = self._numpy_rolling_mean(typical_price, 20)
                mean_deviation = self._numpy_rolling_mean(np.abs(typical_price - sma_tp_20), 20)
                features['cci_20'] = (typical_price - sma_tp_20) / (0.015 * mean_deviation + 1e-10)

                # Money Flow Index (MFI) - if volume available
                if 'volume' in data.columns:
                    raw_money_flow = typical_price * volume_values
                    positive_flow = np.where(typical_price > np.pad(typical_price[:-1], (1, 0), constant_values=typical_price[0]), raw_money_flow, 0)
                    negative_flow = np.where(typical_price < np.pad(typical_price[:-1], (1, 0), constant_values=typical_price[0]), raw_money_flow, 0)
                    
                    positive_mf_14 = self._numpy_rolling_mean(positive_flow, 14)
                    negative_mf_14 = self._numpy_rolling_mean(negative_flow, 14)
                    
                    money_ratio = positive_mf_14 / (negative_mf_14 + 1e-10)
                    features['mfi_14'] = 100 - (100 / (1 + money_ratio))

                # Additional volatility features
                features['volatility_ratio_20_50'] = features['volatility_20'] / (features['volatility_50'] + 1e-10)
                features['volatility_zscore_20'] = (returns - self._numpy_rolling_mean(returns, 20)) / (features['volatility_20'] + 1e-10)

                # Price position indicators
                highest_20 = self._numpy_rolling_max(close_values, 20)
                lowest_20 = self._numpy_rolling_min(close_values, 20)
                features['price_position_20'] = (close_values - lowest_20) / (highest_20 - lowest_20 + 1e-10)

                highest_50 = self._numpy_rolling_max(close_values, 50)
                lowest_50 = self._numpy_rolling_min(close_values, 50)
                features['price_position_50'] = (close_values - lowest_50) / (highest_50 - lowest_50 + 1e-10)

                # Moving average crossovers
                features['sma_ratio_20_50'] = rolling_cache['sma_20'] / (rolling_cache['sma_50'] + 1e-10)
                
                # EMA crossovers
                ema_5 = self._calculate_ema(np.asarray(close_values), 5)
                features['ema_ratio_5_20'] = ema_5 / (rolling_cache['sma_20'] + 1e-10)
                features['ema_ratio_12_26'] = ema_12 / (ema_26 + 1e-10)

                # Rate of change of momentum
                momentum_change_5 = np.diff(momentum_5) / momentum_5[:-1]
                momentum_change_5 = np.pad(momentum_change_5, (1, 0), constant_values=np.nan)
                features['momentum_acceleration_5'] = momentum_change_5[:data_length] if len(momentum_change_5) > data_length else np.pad(momentum_change_5, (0, data_length - len(momentum_change_5)), constant_values=np.nan)

                # Volume Weighted Average Price (VWAP) features
                if 'volume' in data.columns:
                    vwap_5 = self._calculate_vwap(close_values, volume_values, 5)
                    vwap_20 = self._calculate_vwap(close_values, volume_values, 20)
                    features['price_vwap_ratio_5'] = close_values / (vwap_5 + 1e-10)
                    features['price_vwap_ratio_20'] = close_values / (vwap_20 + 1e-10)

                # Additional correlation features
                features['returns_autocorr_1'] = self._calculate_rolling_autocorr(returns, 1)
                features['returns_autocorr_5'] = self._calculate_rolling_autocorr(returns, 5)
                
                # Additional oscillators and indicators
                # ADX-like trend strength
                features['adx_14'] = self._calculate_adx(close_values, high_values, low_values, 14) if all(col in data.columns for col in ['high', 'low']) else np.nan
                
                # Parabolic SAR approximation
                features['sar_trend'] = self._calculate_sar_trend(close_values, high_values, low_values) if all(col in data.columns for col in ['high', 'low']) else np.nan
                
                # Ichimoku Cloud components (simplified)
                features['ichimoku_tenkan'] = (self._numpy_rolling_max(close_values, 9) + self._numpy_rolling_min(close_values, 9)) / 2
                features['ichimoku_kijun'] = (self._numpy_rolling_max(close_values, 26) + self._numpy_rolling_min(close_values, 26)) / 2
                
                # Elder's Force Index
                if 'volume' in data.columns:
                    features['efi_13'] = self._numpy_rolling_mean(returns * volume_values, 13)
                
                # Chaikin Oscillator
                if 'volume' in data.columns and all(col in data.columns for col in ['high', 'low']):
                    ad_line = self._calculate_ad_line(close_values, high_values, low_values, volume_values)
                    features['chaikin_osc'] = self._calculate_ema(ad_line, 3) - self._calculate_ema(ad_line, 10)
                
                # Detrended Price Oscillator
                features['dpo_20'] = close_values - self._numpy_rolling_mean(close_values, 20)[11:]  # Shifted by (20/2 + 1)
                features['dpo_20'] = np.pad(features['dpo_20'], (11, 0), constant_values=np.nan)[:data_length]
                
                # KST (Know Sure Thing)
                roc1 = self._calculate_roc(close_values, 10)
                roc2 = self._calculate_roc(close_values, 15)
                roc3 = self._calculate_roc(close_values, 20)
                roc4 = self._calculate_roc(close_values, 30)
                features['kst'] = (self._calculate_ema(roc1, 10) + self._calculate_ema(roc2, 10) + 
                                  self._calculate_ema(roc3, 10) + self._calculate_ema(roc4, 15)) / 4
                features['kst_signal'] = self._calculate_ema(features['kst'], 9)
                
                # Ultimate Oscillator
                if all(col in data.columns for col in ['high', 'low']) and 'volume' in data.columns:
                    features['ultimate_osc'] = self._calculate_ultimate_oscillator(close_values, high_values, low_values, volume_values)
                
                # More volatility features
                features['volatility_skew_20'] = self._calculate_rolling_skew(returns, 20)
                features['volatility_kurt_20'] = self._calculate_rolling_kurt(returns, 20)
                
                # Price efficiency ratio
                features['efficiency_ratio_14'] = self._calculate_efficiency_ratio(close_values, 14)
                
                # Fractal Dimension Index
                features['fractal_dim_50'] = self._calculate_fractal_dimension(close_values, 50)
                
                # Hurst Exponent
                features['hurst_50'] = self._calculate_hurst_exponent(close_values, 50)
                
                # Multi-timeframe features (if enabled)
                if self.enable_mtf_features:
                    # 4h resampled features (approximate by using larger windows)
                    features['volatility_100'] = self._numpy_rolling_std(returns, 100)
                    features['trend_strength_100'] = (close_values - np.pad(close_values[:-100], (100, 0), constant_values=close_values[0])) / close_values
                    features['trend_strength_100'] = features['trend_strength_100'][:data_length] if len(features['trend_strength_100']) > data_length else np.pad(features['trend_strength_100'], (0, data_length - len(features['trend_strength_100'])), constant_values=np.nan)
                    
                    # Daily context features (approximate by using even larger windows)
                    features['volatility_200'] = self._numpy_rolling_std(returns, 200)
                    features['trend_strength_200'] = (close_values - np.pad(close_values[:-200], (200, 0), constant_values=close_values[0])) / close_values
                    features['trend_strength_200'] = features['trend_strength_200'][:data_length] if len(features['trend_strength_200']) > data_length else np.pad(features['trend_strength_200'], (0, data_length - len(features['trend_strength_200'])), constant_values=np.nan)

            # Ensure all features have the same length as the original data
            final_features = {}
            for key, value in features.items():
                if len(value) != data_length:
                    if len(value) > data_length:
                        final_features[key] = value[:data_length]
                    else:
                        final_features[key] = np.pad(value, (0, data_length - len(value)), constant_values=np.nan)
                else:
                    final_features[key] = value

            features_df = pd.DataFrame(final_features, index=data.index[:data_length])
            
            # Add long window features if needed to reach minimum
            if len(features_df.columns) < self.min_features:
                tprint_info(f"   📈 Adding long-window features to reach minimum {self.min_features}")
                long_window_features = self._add_long_window_features(data, features_df)
                if long_window_features is not None and len(long_window_features.columns) > 0:
                    features_df = pd.concat([features_df, long_window_features], axis=1)
                    tprint_success(f"   ✅ Added {len(long_window_features.columns)} long-window features")

            return features_df
            
        except Exception as e:
            raise Exception(f"Statistical optimizer failed: {e}")
    
    def _generate_features_with_regime_generators(self, data: pd.DataFrame, generators) -> pd.DataFrame:
        """Generate features using the existing comprehensive regime feature generators."""
        try:
            all_features = {}
            feature_count = 0
            
            tprint_info("   🎯 Using comprehensive regime feature generators:")
            
            for i, generator in enumerate(generators):
                try:
                    generator_name = generator.__class__.__name__
                    tprint_info(f"      Generator {i+1}/{len(generators)}: {generator_name}")
                    
                    # Check if generator has generate_features method (FeatureGenerator)
                    if hasattr(generator, 'generate_features'):
                        tprint_info(f"         📋 Using generate_features method")
                        try:
                            gen_features = generator.generate_features(data)
                            tprint_info(f"         📊 Generated features type: {type(gen_features)}")
                        except Exception as gen_error:
                            tprint_warning(f"         ❌ generate_features failed: {gen_error}")
                            continue
                    else:
                        # Use generate method and extract features from FeatureResult (VectorizedFeatureGenerator)
                        tprint_info(f"         🔄 Using generate method (VectorizedFeatureGenerator)")
                        try:
                            result = generator.generate(data)
                            if result:
                                tprint_info(f"         📊 Result type: {type(result)}")
                                if hasattr(result, 'features') and result.features is not None:
                                    gen_features = result.features
                                    tprint_info(f"         📊 Extracted features from result.features: {type(gen_features)}")
                                elif hasattr(result, 'data') and result.data is not None:
                                    gen_features = result.data
                                    tprint_info(f"         📊 Extracted features from result.data: {type(gen_features)}")
                                else:
                                    tprint_warning(f"         ⚠️ No features found in result - available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                                    continue
                            else:
                                tprint_warning(f"         ⚠️ No result returned from generate method")
                                continue
                        except Exception as gen_error:
                            tprint_warning(f"         ❌ generate method failed: {gen_error}")
                            continue

                    # Validate and process generated features
                    if gen_features is not None:
                        if isinstance(gen_features, dict):
                            feature_len = len(gen_features)
                        elif isinstance(gen_features, pd.DataFrame):
                            feature_len = len(gen_features.columns)
                        elif isinstance(gen_features, np.ndarray):
                            feature_len = gen_features.shape[1] if gen_features.ndim > 1 else 1
                        else:
                            feature_len = 0
                            
                        tprint_info(f"         📏 Feature count: {feature_len}")
                        
                        if feature_len > 0:
                            try:
                                # If features is a dict, update directly
                                if isinstance(gen_features, dict):
                                    all_features.update(gen_features)
                                    feature_count += len(gen_features)
                                # If features is a DataFrame, convert to dict
                                elif isinstance(gen_features, pd.DataFrame):
                                    all_features.update(gen_features.to_dict('series'))
                                    feature_count += len(gen_features.columns)
                                # If features is a numpy array, create generic names
                                elif isinstance(gen_features, np.ndarray):
                                    if gen_features.ndim == 1:
                                        all_features[f'feature_{feature_count}'] = gen_features
                                        feature_count += 1
                                    elif gen_features.ndim == 2:
                                        for j in range(gen_features.shape[1]):
                                            all_features[f'{generator_name.lower()}_feature_{j}'] = gen_features[:, j]
                                            feature_count += 1
                                
                                tprint_success(f"         ✓ Successfully added {feature_len} features")
                            except Exception as proc_error:
                                tprint_warning(f"         ❌ Error processing features: {proc_error}")
                                continue
                        else:
                            tprint_warning(f"         ⚠️ No features generated (empty result)")
                            # Try to generate basic features as fallback
                            try:
                                tprint_info(f"         🔄 Attempting to generate basic features as fallback for {generator_name}")
                                basic_features = self._generate_basic_fallback_features(data, generator_name)
                                if basic_features is not None and len(basic_features) > 0:
                                    all_features.update(basic_features)
                                    feature_count += len(basic_features)
                                    tprint_success(f"         ✅ Generated {len(basic_features)} basic fallback features")
                            except Exception as fallback_error:
                                tprint_warning(f"         ❌ Basic fallback failed: {fallback_error}")
                    else:
                        tprint_warning(f"         ⚠️ No features generated (None result)")
                        # Try to generate basic features as fallback
                        try:
                            tprint_info(f"         🔄 Attempting to generate basic features as fallback for {generator_name}")
                            basic_features = self._generate_basic_fallback_features(data, generator_name)
                            if basic_features is not None and len(basic_features) > 0:
                                all_features.update(basic_features)
                                feature_count += len(basic_features)
                                tprint_success(f"         ✅ Generated {len(basic_features)} basic fallback features")
                        except Exception as fallback_error:
                            tprint_warning(f"         ❌ Basic fallback failed: {fallback_error}")
                        
                except AttributeError as ae:
                    if "generate_features" in str(ae):
                        tprint_warning(f"      ⚠️ Generator {generator.__class__.__name__} does not have generate_features method - skipping")
                    else:
                        tprint_warning(f"      ❌ Generator {generator.__class__.__name__} attribute error: {ae}")
                    continue
                except Exception as e:
                    tprint_warning(f"      ❌ Generator {generator.__class__.__name__} failed: {e}")
                    import traceback
                    tprint_warning(f"         🔍 Traceback: {traceback.format_exc()}")
                    continue
            
            if not all_features:
                raise Exception("No features generated by any regime generator")
            
            # Convert to DataFrame
            features_df = pd.DataFrame(all_features, index=data.index[:len(next(iter(all_features.values())))])
            
            # Remove any constant or NaN features
            features_df = features_df.loc[:, features_df.nunique() > 1]  # Remove constant features
            features_df = features_df.dropna(axis=1, how='all')  # Remove all-NaN columns
            
            # Add important long rolling window features that might be missing from regime generators
            additional_features = self._add_long_window_features(data, features_df)
            if additional_features is not None and len(additional_features.columns) > 0:
                # Combine with existing features
                features_df = pd.concat([features_df, additional_features], axis=1)
                tprint_info(f"   📈 Added {len(additional_features.columns)} long-window features")
            
            tprint_success(f"   ✅ Total features generated: {len(features_df.columns)} (after filtering and additions)")
            
            return features_df
            
        except Exception as e:
            raise Exception(f"Regime generators failed: {e}")
    
    def _add_long_window_features(self, data: pd.DataFrame, existing_features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add important long rolling window features that might be missing from regime generators."""
        try:
            additional_features = {}
            data_length = len(data)
            
            # Get basic price data
            if 'close' in data.columns:
                close_values = data['close'].values
                
                # Calculate returns if not already present
                if 'returns' not in existing_features.columns:
                    returns = np.diff(close_values) / close_values[:-1]
                    returns = np.pad(returns, (1, 0), constant_values=np.nan)
                    if len(returns) != data_length:
                        returns = returns[:data_length] if len(returns) > data_length else np.pad(returns, (0, data_length - len(returns)), constant_values=np.nan)
                else:
                    returns = existing_features['returns'].values
                
                # Long-term volatility features (important for regime detection)
                if 'volatility_12' not in existing_features.columns:
                    additional_features['volatility_12'] = self._numpy_rolling_std(returns, 12)
                
                if 'volatility_26' not in existing_features.columns:
                    additional_features['volatility_26'] = self._numpy_rolling_std(returns, 26)
                
                if 'volatility_72' not in existing_features.columns:
                    additional_features['volatility_72'] = self._numpy_rolling_std(returns, 72)
                
                # Long-term trend strength features
                if 'trend_strength_12' not in existing_features.columns:
                    trend_12 = (close_values - np.pad(close_values[:-12], (12, 0), constant_values=close_values[0])) / close_values
                    additional_features['trend_strength_12'] = trend_12[:data_length] if len(trend_12) > data_length else np.pad(trend_12, (0, data_length - len(trend_12)), constant_values=np.nan)
                
                if 'trend_strength_26' not in existing_features.columns:
                    trend_26 = (close_values - np.pad(close_values[:-26], (26, 0), constant_values=close_values[0])) / close_values
                    additional_features['trend_strength_26'] = trend_26[:data_length] if len(trend_26) > data_length else np.pad(trend_26, (0, data_length - len(trend_26)), constant_values=np.nan)
                
                if 'trend_strength_72' not in existing_features.columns:
                    trend_72 = (close_values - np.pad(close_values[:-72], (72, 0), constant_values=close_values[0])) / close_values
                    additional_features['trend_strength_72'] = trend_72[:data_length] if len(trend_72) > data_length else np.pad(trend_72, (0, data_length - len(trend_72)), constant_values=np.nan)
                
                # Additional long-term features if multi-timeframe is enabled
                if self.enable_mtf_features:
                    # Intermediate-term moving averages for trend context
                    if 'sma_12' not in existing_features.columns:
                        additional_features['sma_12'] = self._numpy_rolling_mean(close_values, 12)
                    
                    if 'sma_26' not in existing_features.columns:
                        additional_features['sma_26'] = self._numpy_rolling_mean(close_values, 26)
                    
                    if 'sma_72' not in existing_features.columns:
                        additional_features['sma_72'] = self._numpy_rolling_mean(close_values, 72)
                    
                    # Price position relative to intermediate-term context
                    if 'price_position_12' not in existing_features.columns:
                        highest_12 = self._numpy_rolling_max(close_values, 12)
                        lowest_12 = self._numpy_rolling_min(close_values, 12)
                        additional_features['price_position_12'] = (close_values - lowest_12) / (highest_12 - lowest_12 + 1e-10)
                    
                    if 'price_position_26' not in existing_features.columns:
                        highest_26 = self._numpy_rolling_max(close_values, 26)
                        lowest_26 = self._numpy_rolling_min(close_values, 26)
                        additional_features['price_position_26'] = (close_values - lowest_26) / (highest_26 - lowest_26 + 1e-10)
                    
                    if 'price_position_72' not in existing_features.columns:
                        highest_72 = self._numpy_rolling_max(close_values, 72)
                        lowest_72 = self._numpy_rolling_min(close_values, 72)
                        additional_features['price_position_72'] = (close_values - lowest_72) / (highest_72 - lowest_72 + 1e-10)
                    
                    # Intermediate-term volatility ratios
                    if 'volatility_ratio_12_26' not in existing_features.columns:
                        vol_12 = additional_features.get('volatility_12', self._numpy_rolling_std(returns, 12))
                        vol_26 = additional_features.get('volatility_26', self._numpy_rolling_std(returns, 26))
                        additional_features['volatility_ratio_12_26'] = vol_12 / (vol_26 + 1e-10)
                    
                    if 'volatility_ratio_26_72' not in existing_features.columns:
                        vol_26 = additional_features.get('volatility_26', self._numpy_rolling_std(returns, 26))
                        vol_72 = additional_features.get('volatility_72', self._numpy_rolling_std(returns, 72))
                        additional_features['volatility_ratio_26_72'] = vol_26 / (vol_72 + 1e-10)
                
                # Volume-based intermediate-term features
                if 'volume' in data.columns:
                    volume_values = data['volume'].values
                    
                    # Intermediate-term volume trends
                    if 'volume_trend_12' not in existing_features.columns:
                        vol_trend_12 = volume_values / np.pad(volume_values[:-12], (12, 0), constant_values=volume_values[0])
                        additional_features['volume_trend_12'] = vol_trend_12[:data_length] if len(vol_trend_12) > data_length else np.pad(vol_trend_12, (0, data_length - len(vol_trend_12)), constant_values=np.nan)
                    
                    if 'volume_trend_26' not in existing_features.columns:
                        vol_trend_26 = volume_values / np.pad(volume_values[:-26], (26, 0), constant_values=volume_values[0])
                        additional_features['volume_trend_26'] = vol_trend_26[:data_length] if len(vol_trend_26) > data_length else np.pad(vol_trend_26, (0, data_length - len(vol_trend_26)), constant_values=np.nan)
                    
                    if 'volume_trend_72' not in existing_features.columns:
                        vol_trend_72 = volume_values / np.pad(volume_values[:-72], (72, 0), constant_values=volume_values[0])
                        additional_features['volume_trend_72'] = vol_trend_72[:data_length] if len(vol_trend_72) > data_length else np.pad(vol_trend_72, (0, data_length - len(vol_trend_72)), constant_values=np.nan)
            
            if additional_features:
                # Ensure all features have correct length
                for key, value in additional_features.items():
                    if len(value) != data_length:
                        if len(value) > data_length:
                            additional_features[key] = value[:data_length]
                        else:
                            additional_features[key] = np.pad(value, (0, data_length - len(value)), constant_values=np.nan)
                
                return pd.DataFrame(additional_features, index=data.index[:data_length])
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to add long-window features: {e}")
            return None
    
    # Helper methods for enhanced technical indicators (kept for backward compatibility)
    def _calculate_adx(self, close, high, low, period=14):
        """Calculate ADX-like trend strength indicator."""
        try:
            tr1 = high - low
            tr2 = np.abs(high - np.pad(close[:-1], (1, 0), constant_values=close[0]))
            tr3 = np.abs(low - np.pad(close[:-1], (1, 0), constant_values=close[0]))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            plus_dm = np.where(high > np.pad(high[:-1], (1, 0), constant_values=high[0]), 
                              high - np.pad(high[:-1], (1, 0), constant_values=high[0]), 0)
            minus_dm = np.where(np.pad(low[:-1], (1, 0), constant_values=low[0]) > low,
                               np.pad(low[:-1], (1, 0), constant_values=low[0]) - low, 0)
            
            tr_smooth = self._calculate_ema(tr, period)
            plus_dm_smooth = self._calculate_ema(plus_dm, period)
            minus_dm_smooth = self._calculate_ema(minus_dm, period)
            
            plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
            minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = self._calculate_ema(dx, period)
            
            return adx
        except:
            return np.full(len(close), np.nan)
    
    def _calculate_sar_trend(self, close, high, low):
        """Calculate simplified Parabolic SAR trend indicator."""
        try:
            # Simplified SAR - just return trend direction based on price momentum
            returns = np.diff(close) / close[:-1]
            momentum = self._numpy_rolling_mean(np.pad(returns, (1, 0), constant_values=0), 5)
            return np.where(momentum > 0, 1, -1)
        except:
            return np.full(len(close), np.nan)
    
    def _calculate_ad_line(self, close, high, low, volume):
        """Calculate Accumulation/Distribution Line."""
        try:
            clv = ((close - low) - (high - close)) / (high - low + 1e-10)
            clv = clv * volume
            return np.cumsum(clv)
        except:
            return np.full(len(close), np.nan)
    
    def _calculate_roc(self, data, period):
        """Calculate Rate of Change."""
        try:
            return ((data - np.pad(data[:-period], (period, 0), constant_values=data[0])) / 
                   (np.pad(data[:-period], (period, 0), constant_values=data[0]) + 1e-10)) * 100
        except:
            return np.full(len(data), np.nan)
    
    def _calculate_ultimate_oscillator(self, close, high, low, volume):
        """Calculate Ultimate Oscillator."""
        try:
            bp = close - np.minimum(low, np.pad(close[:-1], (1, 0), constant_values=close[0]))
            tr = np.maximum(high, np.pad(close[:-1], (1, 0), constant_values=close[0])) - \
                 np.minimum(low, np.pad(close[:-1], (1, 0), constant_values=close[0]))
            
            avg7 = self._numpy_rolling_mean(bp, 7) / (self._numpy_rolling_mean(tr, 7) + 1e-10)
            avg14 = self._numpy_rolling_mean(bp, 14) / (self._numpy_rolling_mean(tr, 14) + 1e-10)
            avg28 = self._numpy_rolling_mean(bp, 28) / (self._numpy_rolling_mean(tr, 28) + 1e-10)
            
            uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
            return uo
        except:
            return np.full(len(close), np.nan)
    
    def _calculate_rolling_skew(self, data, window):
        """Calculate rolling skewness."""
        try:
            result = np.full(len(data), np.nan)
            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if len(window_data) == window and np.all(np.isfinite(window_data)):
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    if std_val > 0:
                        result[i] = np.mean(((window_data - mean_val) / std_val) ** 3)
            return result
        except:
            return np.full(len(data), np.nan)
    
    def _calculate_rolling_kurt(self, data, window):
        """Calculate rolling kurtosis."""
        try:
            result = np.full(len(data), np.nan)
            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if len(window_data) == window and np.all(np.isfinite(window_data)):
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    if std_val > 0:
                        result[i] = np.mean(((window_data - mean_val) / std_val) ** 4) - 3
            return result
        except:
            return np.full(len(data), np.nan)
    
    def _calculate_efficiency_ratio(self, data, period):
        """Calculate Efficiency Ratio (from Perry Kaufman)."""
        try:
            direction = np.abs(data - np.pad(data[:-period], (period, 0), constant_values=data[0]))
            volatility = np.sum(np.abs(np.diff(data))[max(0, period-1):], axis=0)
            volatility = np.pad(volatility, (period-1, 0), constant_values=0)
            return direction / (volatility + 1e-10)
        except:
            return np.full(len(data), np.nan)
    
    def _calculate_fractal_dimension(self, data, window):
        """Calculate Fractal Dimension Index."""
        try:
            result = np.full(len(data), np.nan)
            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if len(window_data) == window and np.all(np.isfinite(window_data)):
                    # Simplified fractal dimension calculation
                    max_val = np.max(window_data)
                    min_val = np.min(window_data)
                    if max_val > min_val:
                        normalized = (window_data - min_val) / (max_val - min_val)
                        # Count number of direction changes
                        changes = np.sum(np.diff(np.sign(np.diff(normalized))) != 0)
                        result[i] = 1 + np.log(changes + 1) / np.log(window)
            return result
        except:
            return np.full(len(data), np.nan)
    
    def _calculate_hurst_exponent(self, data, window):
        """Calculate simplified Hurst Exponent."""
        try:
            result = np.full(len(data), np.nan)
            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if len(window_data) == window and np.all(np.isfinite(window_data)):
                    # Simplified Hurst calculation using rescaled range
                    mean_val = np.mean(window_data)
                    deviates = window_data - mean_val
                    cumsum = np.cumsum(deviates)
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(window_data)
                    if s > 0:
                        result[i] = np.log(r / s) / np.log(window)
            return result
        except:
            return np.full(len(data), np.nan)
    
    def _generate_features_fallback(self, data: pd.DataFrame, generators) -> pd.DataFrame:
        """Fallback feature generation using individual generators."""
        try:
            all_features = {}
            for generator in generators:
                try:
                    # Check if generator has generate_features method (FeatureGenerator)
                    if hasattr(generator, 'generate_features'):
                        gen_features = generator.generate_features(data)
                    else:
                        # Use generate method and extract features from FeatureResult (VectorizedFeatureGenerator)
                        result = generator.generate(data)
                        if result and hasattr(result, 'features'):
                            gen_features = result.features
                        else:
                            gen_features = None

                    if gen_features is not None:
                        # If features is a dict, update directly
                        if isinstance(gen_features, dict):
                            all_features.update(gen_features)
                        # If features is a Series, add with appropriate name
                        elif isinstance(gen_features, pd.Series):
                            all_features[generator.config.name] = gen_features
                except Exception as e:
                    tprint_warning(f"   ⚠️ Generator {generator.__class__.__name__} failed: {e}")

            # Convert to DataFrame
            features_df = pd.DataFrame(all_features, index=data.index)
            tprint_info(f"   Generated {features_df.shape[1]} features from individual generators")
            return features_df
            
        except Exception as e:
            raise Exception(f"Fallback generation failed: {e}")
    
    def _numpy_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized numpy rolling mean using convolution."""
        # Ensure data is a numpy array
        data_array = np.asarray(data)
        result = np.full(len(data_array), np.nan)
        if len(data_array) >= window:
            cumsum = np.cumsum(np.insert(data_array, 0, 0))
            result[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
        return result
    
    def _numpy_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized numpy rolling standard deviation."""
        # Ensure data is a numpy array
        data_array = np.asarray(data)
        result = np.full(len(data_array), np.nan)
        if len(data_array) >= window:
            # Use rolling mean and then compute std
            rolling_mean = self._numpy_rolling_mean(data_array, window)
            cumsum2 = np.cumsum(np.insert(data_array**2, 0, 0))
            rolling_sq_mean = (cumsum2[window:] - cumsum2[:-window]) / window
            # Ensure both arrays have the same length before broadcasting
            rolling_mean_windowed = rolling_mean[window-1:]
            if len(rolling_mean_windowed) == len(rolling_sq_mean):
                result[window-1:] = np.sqrt(rolling_sq_mean - rolling_mean_windowed**2)
            else:
                # Fallback: calculate std manually if lengths don't match
                min_len = min(len(rolling_mean_windowed), len(rolling_sq_mean))
                result[window-1:window-1+min_len] = np.sqrt(rolling_sq_mean[:min_len] - rolling_mean_windowed[:min_len]**2)
        return result

    def _calculate_ema(self, data: np.ndarray, span: int) -> np.ndarray:
        """Calculate exponential moving average."""
        data_array = np.asarray(data)
        alpha = 2 / (span + 1)
        ema = np.full(len(data_array), np.nan)
        ema[0] = data_array[0]
        for i in range(1, len(data_array)):
            ema[i] = alpha * data_array[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_rolling_corr(self, x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling correlation between two arrays."""
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        result = np.full(len(x_array), np.nan)
        if len(x_array) >= window:
            for i in range(window-1, len(x_array)):
                x_window = x_array[i-window+1:i+1]
                y_window = y_array[i-window+1:i+1]
                if not np.isnan(x_window).any() and not np.isnan(y_window).any():
                    result[i] = np.corrcoef(x_window, y_window)[0, 1]
        return result
    
    def _numpy_rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized numpy rolling maximum."""
        data_array = np.asarray(data)
        result = np.full(len(data_array), np.nan)
        if len(data_array) >= window:
            for i in range(window-1, len(data_array)):
                result[i] = np.nanmax(data_array[i-window+1:i+1])
        return result
    
    def _numpy_rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized numpy rolling minimum."""
        data_array = np.asarray(data)
        result = np.full(len(data_array), np.nan)
        if len(data_array) >= window:
            for i in range(window-1, len(data_array)):
                result[i] = np.nanmin(data_array[i-window+1:i+1])
        return result
    
    def _calculate_vwap(self, price: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate Volume Weighted Average Price (VWAP)."""
        price_array = np.asarray(price)
        volume_array = np.asarray(volume)
        result = np.full(len(price_array), np.nan)
        if len(price_array) >= window:
            for i in range(window-1, len(price_array)):
                price_window = price_array[i-window+1:i+1]
                volume_window = volume_array[i-window+1:i+1]
                if not np.isnan(price_window).any() and not np.isnan(volume_window).any() and np.sum(volume_window) > 0:
                    result[i] = np.sum(price_window * volume_window) / np.sum(volume_window)
        return result
    
    def _calculate_rolling_autocorr(self, data: np.ndarray, lag: int) -> np.ndarray:
        """Calculate rolling autocorrelation with given lag."""
        data_array = np.asarray(data)
        result = np.full(len(data_array), np.nan)
        window = 20  # Use 20-period window for autocorrelation
        if len(data_array) >= window + lag:
            for i in range(window + lag - 1, len(data_array)):
                current_window = data_array[i-window+1:i+1]
                lagged_window = data_array[i-window+1-lag:i+1-lag]
                if not np.isnan(current_window).any() and not np.isnan(lagged_window).any():
                    correlation = np.corrcoef(current_window, lagged_window)[0, 1]
                    result[i] = correlation if not np.isnan(correlation) else 0.0
        return result
    
    def _generate_basic_fallback_features(self, data: pd.DataFrame, generator_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate basic fallback features when a generator fails.
        
        This ensures we always have some features even when specialized generators fail.
        
        Args:
            data: Market data DataFrame
            generator_name: Name of the failed generator for logging
            
        Returns:
            Dictionary of basic features or None if generation fails
        """
        try:
            features = {}
            data_length = len(data)
            
            # Basic price features
            if 'close' in data.columns:
                close_values = data['close'].values
                
                # Simple returns
                returns = np.diff(close_values) / close_values[:-1]
                returns = np.pad(returns, (1, 0), constant_values=np.nan)
                if len(returns) != data_length:
                    returns = returns[:data_length] if len(returns) > data_length else np.pad(returns, (0, data_length - len(returns)), constant_values=np.nan)
                
                features[f'{generator_name.lower()}_returns'] = returns
                features[f'{generator_name.lower()}_log_returns'] = np.log(close_values / np.pad(close_values[:-1], (1, 0), constant_values=close_values[0]))
                
                # Simple moving averages
                if data_length >= 5:
                    features[f'{generator_name.lower()}_sma_5'] = self._numpy_rolling_mean(close_values, 5)
                if data_length >= 10:
                    features[f'{generator_name.lower()}_sma_10'] = self._numpy_rolling_mean(close_values, 10)
                if data_length >= 20:
                    features[f'{generator_name.lower()}_sma_20'] = self._numpy_rolling_mean(close_values, 20)
                
                # Price ratios
                if data_length >= 5:
                    sma_5 = self._numpy_rolling_mean(close_values, 5)
                    features[f'{generator_name.lower()}_price_sma_5_ratio'] = close_values / (sma_5 + 1e-10)
                
                # Simple volatility
                if len(returns) >= 10:
                    features[f'{generator_name.lower()}_volatility_10'] = self._numpy_rolling_std(returns, 10)
                if len(returns) >= 20:
                    features[f'{generator_name.lower()}_volatility_20'] = self._numpy_rolling_std(returns, 20)
            
            # Basic volume features
            if 'volume' in data.columns:
                volume_values = data['volume'].values
                
                # Volume change
                volume_change = np.diff(volume_values) / volume_values[:-1]
                volume_change = np.pad(volume_change, (1, 0), constant_values=np.nan)
                if len(volume_change) != data_length:
                    volume_change = volume_change[:data_length] if len(volume_change) > data_length else np.pad(volume_change, (0, data_length - len(volume_change)), constant_values=np.nan)
                
                features[f'{generator_name.lower()}_volume_change'] = volume_change
                
                # Volume SMA
                if data_length >= 10:
                    features[f'{generator_name.lower()}_volume_sma_10'] = self._numpy_rolling_mean(volume_values, 10)
                
                # Volume ratio
                if data_length >= 10:
                    volume_sma_10 = self._numpy_rolling_mean(volume_values, 10)
                    features[f'{generator_name.lower()}_volume_ratio'] = volume_values / (volume_sma_10 + 1e-10)
            
            # Basic OHLC features
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                open_values = data['open'].values
                high_values = data['high'].values
                low_values = data['low'].values
                close_values = data['close'].values
                
                # Price range features
                features[f'{generator_name.lower()}_high_low_ratio'] = (high_values - low_values) / (close_values + 1e-10)
                features[f'{generator_name.lower()}_open_close_ratio'] = (open_values - close_values) / (close_values + 1e-10)
                
                # Upper/lower shadow ratios
                body_size = np.abs(close_values - open_values)
                upper_shadow = high_values - np.maximum(open_values, close_values)
                lower_shadow = np.minimum(open_values, close_values) - low_values
                
                features[f'{generator_name.lower()}_upper_shadow_ratio'] = upper_shadow / (body_size + 1e-10)
                features[f'{generator_name.lower()}_lower_shadow_ratio'] = lower_shadow / (body_size + 1e-10)
            
            # Ensure all features have correct length
            for key, value in features.items():
                if len(value) != data_length:
                    if len(value) > data_length:
                        features[key] = value[:data_length]
                    else:
                        features[key] = np.pad(value, (0, data_length - len(value)), constant_values=np.nan)
            
            return features if features else None
            
        except Exception as e:
            tprint_warning(f"   ❌ Basic fallback feature generation failed for {generator_name}: {e}")
            return None
    
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
        
        # Handle excessive NaN values - IMPROVED APPROACH
        tprint_info("🔍 Checking for excessive NaN values...")
        nan_ratio = feature_matrix.isna().sum().sum() / (feature_matrix.shape[0] * feature_matrix.shape[1])
        tprint_info(f"   Overall NaN ratio: {nan_ratio:.1%}")
        
        # Show detailed NaN analysis for debugging
        nan_per_column = feature_matrix.isna().sum() / len(feature_matrix)
        high_nan_features = nan_per_column[nan_per_column > 0.01].sort_values(ascending=False)
        
        if len(high_nan_features) > 0:
            tprint_warning(f"   ⚠️ Found {len(high_nan_features)} features with >1% NaN values:")
            for feature, nan_pct in high_nan_features.head(10).items():
                window_info = ""
                if '_20' in feature:
                    window_info = " (20-day window)"
                elif '_50' in feature:
                    window_info = " (50-day window)"
                tprint_info(f"      • {feature}: {nan_pct:.1%} NaN{window_info}")

        if nan_ratio > 0.15:  # Increased threshold for short timeframes
            tprint_warning(f"   ⚠️ High NaN ratio detected: {nan_ratio:.1%}")

            # IMPROVED: Check individual columns but be more lenient for short timeframes
            # For 60-day data, we expect higher NaN ratios for longer windows
            problematic_columns = []
            for col in feature_matrix.columns:
                col_nan_ratio = feature_matrix[col].isna().sum() / len(feature_matrix)
                # More lenient threshold for short datasets
                if col_nan_ratio > 0.95:  # Only remove if >95% NaN
                    problematic_columns.append((col, col_nan_ratio))
                    tprint_warning(f"      Column '{col}' has {col_nan_ratio:.1%} NaN values (will remove)")
                elif col_nan_ratio > 0.70:  # Warn about very high NaN
                    tprint_info(f"      Column '{col}' has {col_nan_ratio:.1%} NaN values (keeping - insufficient data for window)")

            # Remove only truly problematic columns
            if problematic_columns:
                columns_to_remove = [col for col, _ in problematic_columns]
                tprint_warning(f"   🗑️ Removing {len(columns_to_remove)} columns with >95% NaN values")
                feature_matrix = feature_matrix.drop(columns=columns_to_remove)

                # Update feature names
                feature_names = [f for f in feature_names if f not in columns_to_remove]

            # IMPROVED: Always fill remaining NaN values with forward fill and zero
            # This is appropriate for time series features with rolling windows
            tprint_info("   🔧 Filling remaining NaN values with forward fill and zero (appropriate for time series)")
            feature_matrix = feature_matrix.fillna(method='ffill').fillna(0)
            final_nan_ratio = feature_matrix.isna().sum().sum() / (feature_matrix.shape[0] * feature_matrix.shape[1])
            tprint_info(f"   ✅ Final NaN ratio after intelligent handling: {final_nan_ratio:.1%}")
        
        # Validate we have enough features
        if len(feature_matrix.columns) < 5:
            raise ValueError(f"Too few valid features after NaN handling: {len(feature_matrix.columns)} (minimum: 5)")
        
        tprint_success(f"✅ Using {len(feature_matrix.columns)} features for clustering")
        
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
                feature_matrix,  # Pass DataFrame with column names instead of .values
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
        
        # IMPORTANT: Align feature_matrix with cluster_labels to prevent length mismatch
        # The clustering might return fewer labels due to preprocessing (e.g., NaN removal)
        if len(result.cluster_labels) != len(feature_matrix):
            tprint_warning(f"⚠️ Length mismatch between cluster_labels ({len(result.cluster_labels)}) "
                          f"and feature_matrix ({len(feature_matrix)})")
            # Align feature_matrix to match cluster_labels length
            feature_matrix = feature_matrix.iloc[:len(result.cluster_labels)].reset_index(drop=True)
            tprint_info(f"✅ Aligned feature_matrix to {len(feature_matrix)} samples")
        
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
                'composite_score': result.composite_score,
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
        
        # Run comprehensive quality assessment using ClusterQualityAssessor if available
        if QUALITY_ASSESSOR_AVAILABLE:
            tprint_info("🔍 Running comprehensive quality assessment...")
            try:
                # Initialize quality assessor
                quality_assessor = ClusterQualityAssessor(
                    artifact_manager=None,
                    enable_hardware_optimization=True,
                    enable_vectorization=True
                )
                
                # Extract data for quality assessment
                cluster_labels = result.cluster_labels
                feature_data = feature_matrix
                
                # Ensure data alignment
                min_length = min(len(cluster_labels), len(feature_data))
                cluster_labels = cluster_labels[:min_length]
                feature_data = feature_data.iloc[:min_length].reset_index(drop=True)
                timestamps = data.index[:min_length]
                
                # Calculate forward returns for economic validation
                forward_returns = None
                if 'close' in data.columns:
                    forward_returns = data['close'].pct_change().shift(-1).iloc[:min_length]
                elif 'returns' in data.columns:
                    forward_returns = data['returns'].shift(-1).iloc[:min_length]
                
                # Run comprehensive quality assessment
                comprehensive_quality = quality_assessor.assess_quality(
                    regime_labels=cluster_labels,
                    feature_data=feature_data,
                    forward_returns=forward_returns,
                    timestamps=timestamps,
                    min_regime_size=10,
                    temporal_sensitivity_mode="standard"
                )
                
                # Add comprehensive quality metrics to results
                return_dict['comprehensive_quality_metrics'] = comprehensive_quality.to_dict()
                return_dict['enhanced_quality_score'] = comprehensive_quality.quality_score or 0.0
                
                tprint_success(f"✅ Comprehensive quality assessment: {comprehensive_quality.quality_score:.4f}")
                
                # Save detailed CSV reports
                self._save_comprehensive_quality_reports(comprehensive_quality, data)
                
            except Exception as e:
                tprint_error(f"❌ Comprehensive quality assessment failed: {e}")
                tprint_warning(f"⚠️ Using basic quality metrics only")
        
        return return_dict
    
    def _save_comprehensive_quality_reports(self, quality_metrics: ClusterQualityMetrics, 
                                            data: pd.DataFrame) -> None:
        """
        Save detailed quality assessment reports to CSV files.
        
        Args:
            quality_metrics: ClusterQualityMetrics object
            data: Original market data for context
        """
        tprint_info("💾 Generating comprehensive quality assessment CSV...")
        
        try:
            # Create output directory
            from pathlib import Path
            from datetime import datetime
            
            output_dir = Path("outcomes") / "enhanced_sticky_finite_hmm_quality_reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Main summary report
            summary_data = {
                'Metric': [
                    'Silhouette Score',
                    'Davies-Bouldin Index', 
                    'Calinski-Harabasz Index',
                    'Within-Regime CV',
                    'Between-Regime CV',
                    'Temporal Smoothness',
                    'Regime Persistence',
                    'Number of Regimes',
                    'Noise Ratio',
                    'Balance Score',
                    'Overall Quality Score'
                ],
                'Value': [
                    quality_metrics.silhouette_score,
                    quality_metrics.davies_bouldin_score,
                    quality_metrics.calinski_harabasz_score,
                    quality_metrics.within_regime_cv,
                    quality_metrics.between_regime_cv,
                    quality_metrics.temporal_smoothness,
                    quality_metrics.regime_persistence,
                    quality_metrics.n_regimes,
                    quality_metrics.noise_ratio,
                    quality_metrics.balance_score,
                    quality_metrics.quality_score
                ],
                'Description': [
                    'Cluster separation quality (-1 to 1, higher better)',
                    'Cluster separation quality (lower better)',
                    'Cluster separation quality (higher better)',
                    'Within-regime feature consistency (lower better)',
                    'Between-regime feature separation (higher better)',
                    'Temporal stability of regimes (0 to 1, higher better)',
                    'Average regime duration in periods',
                    'Number of discovered regimes',
                    'Ratio of noise points (lower better)',
                    'Balance of cluster sizes (0 to 1, higher better)',
                    'Composite quality score (0 to 1, higher better)'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = output_dir / f"enhanced_clustering_quality_summary_{timestamp}.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            
            # 2. Per-regime detailed metrics
            if quality_metrics.per_regime_metrics:
                regime_data = []
                for regime_id, regime_metrics in quality_metrics.per_regime_metrics.items():
                    regime_data.append({
                        'Regime_ID': regime_id,
                        'Size': regime_metrics.get('size', 0),
                        'Size_Percentage': regime_metrics.get('size_pct', 0),
                        'Mean_Return': regime_metrics.get('mean_return', 0),
                        'Volatility': regime_metrics.get('volatility', 0),
                        'Sharpe_Ratio': regime_metrics.get('sharpe', 0),
                        'Max_Drawdown': regime_metrics.get('max_drawdown', 0),
                        'Win_Rate': regime_metrics.get('win_rate', 0),
                        'Regime_Type': regime_metrics.get('regime_type', 'unknown'),
                        'Duration_Mean': regime_metrics.get('duration_mean', 0),
                        'Duration_Std': regime_metrics.get('duration_std', 0)
                    })
                
                regime_df = pd.DataFrame(regime_data)
                regime_csv_path = output_dir / f"enhanced_regime_detailed_metrics_{timestamp}.csv"
                regime_df.to_csv(regime_csv_path, index=False)
            
            # 3. Economic validation metrics
            if quality_metrics.economic_validation:
                econ_data = {
                    'Economic_Metric': [
                        'Portfolio Return',
                        'Portfolio Sharpe Ratio',
                        'Max Drawdown',
                        'Volatility',
                        'Hit Rate',
                        'Profit Factor',
                        'Average Trade Return',
                        'Target Return Achievement'
                    ],
                    'Value': [
                        quality_metrics.economic_validation.get('portfolio_return', 0),
                        quality_metrics.economic_validation.get('portfolio_sharpe', 0),
                        quality_metrics.economic_validation.get('max_drawdown', 0),
                        quality_metrics.economic_validation.get('portfolio_volatility', 0),
                        quality_metrics.economic_validation.get('hit_rate', 0),
                        quality_metrics.economic_validation.get('profit_factor', 0),
                        quality_metrics.economic_validation.get('avg_trade_return', 0),
                        quality_metrics.economic_validation.get('target_return_achievement', 0)
                    ],
                    'Benchmark': [
                        'Higher better',
                        'Higher better',
                        'Lower better', 
                        'Lower better',
                        'Higher better',
                        'Higher better',
                        'Higher better',
                        'Higher better'
                    ]
                }
                
                econ_df = pd.DataFrame(econ_data)
                econ_csv_path = output_dir / f"enhanced_economic_validation_{timestamp}.csv"
                econ_df.to_csv(econ_csv_path, index=False)
            
            # 4. Temporal analysis metrics
            temporal_data = {
                'Temporal_Metric': [
                    'Temporal Smoothness',
                    'Temporal Smoothness (Raw)',
                    'Flip-Flop Ratio',
                    'Regime Persistence',
                    'Average Duration',
                    'Duration Std Dev',
                    'Min Duration',
                    'Max Duration'
                ],
                'Value': [
                    quality_metrics.temporal_smoothness,
                    quality_metrics.temporal_smoothness_raw,
                    quality_metrics.flip_flop_ratio,
                    quality_metrics.regime_persistence,
                    quality_metrics.regime_duration_distribution.get('mean_duration', 0),
                    quality_metrics.regime_duration_distribution.get('std_duration', 0),
                    quality_metrics.regime_duration_distribution.get('min_duration', 0),
                    quality_metrics.regime_duration_distribution.get('max_duration', 0)
                ],
                'Interpretation': [
                    'Higher = more stable regimes',
                    'Higher = more stable (no penalty)',
                    'Lower = fewer rapid switches',
                    'Higher = longer lasting regimes',
                    'Average regime length in periods',
                    'Variability in regime duration',
                    'Shortest regime observed',
                    'Longest regime observed'
                ]
            }
            
            temporal_df = pd.DataFrame(temporal_data)
            temporal_csv_path = output_dir / f"enhanced_temporal_analysis_{timestamp}.csv"
            temporal_df.to_csv(temporal_csv_path, index=False)
            
            tprint_success(f"✅ Enhanced CSV reports saved to {output_dir}")
            tprint_info(f"   📄 Summary: {summary_csv_path.name}")
            if quality_metrics.per_regime_metrics:
                tprint_info(f"   📄 Regimes: {regime_csv_path.name}")
            if quality_metrics.economic_validation:
                tprint_info(f"   📄 Economic: {econ_csv_path.name}")
            tprint_info(f"   📄 Temporal: {temporal_csv_path.name}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save enhanced quality reports: {e}")
    
    def clear_cache(self, persistent: bool = False):
        """Clear feature caches.
        
        Args:
            persistent: Whether to clear persistent cache as well
        """
        self.cache_manager.clear_cache(persistent=persistent)
        if persistent:
            self._persistent_feature_cache.clear()
        self._cached_features = None
        self._cached_data_hash = None
        tprint_info(f"🧹 Cleared {'persistent and ' if persistent else ''}instance caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'instance_cache_size': len(getattr(self.cache_manager, '_instance_cache', {})),
            'persistent_cache_size': len(getattr(self.cache_manager, '_persistent_cache', {})),
            'data_hash_cache_size': len(getattr(self.cache_manager, '_data_hash_cache', {})),
            'legacy_persistent_cache_size': len(self._persistent_feature_cache),
            'has_cached_features': self._cached_features is not None
        }


# Convenience function
def perform_enhanced_sticky_finite_hmm_clustering(
    data: pd.DataFrame,
    min_features: int = 30,  # Reduced from 50 to be more realistic
    max_features: int = 100,
    K: int = 5,
    base_alpha: float = 0.5,
    kappa: float = 10.0,
    num_iters: int = 150,  # Reduced from 800 for faster training
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

