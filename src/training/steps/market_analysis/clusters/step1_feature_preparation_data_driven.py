"""
Step 1: Data-Driven Feature Preparation for NAS-TAS Clustering.

This module handles feature selection, dimensionality reduction, and regime-specific
feature integration for the clustering process with data-driven parameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance
)
from src.utils.common_operations import (
    get_memory_usage, optimize_dataframe_memory, safe_divide, safe_mean, safe_std,
    memory_monitor, force_garbage_collection, performance_timer, validate_dataframe,
    safe_merge, safe_concat, calculate_data_quality_metrics, create_summary_statistics,
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes
)
from src.utils.common_utilities import (
    analyze_nan_values_detailed, format_nan_analysis_report, get_dataframe_info,
    safe_merge_dataframes, safe_groupby_operation, safe_apply_function,
    calculate_data_quality_metrics, create_summary_statistics
)
from src.utils.math_validation import (
    validate_finite, validate_array_finite, safe_divide, safe_log, safe_sqrt, safe_power,
    safe_correlation, safe_mean, safe_std, validate_positive, safe_covariance,
    safe_percentile, validate_correlation_matrix
)
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, IntegratedHardwareManager
)
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, VectorizationConfig
)
from src.utils.data.unified_data_utils import (
    UnifiedDataUtils, DataQualityMetrics, DataValidationResult
)

# Import data-driven optimization
try:
    from ..hdbscan_clustering.optimization.data_driven_feature_weights import (
        DataDrivenFeatureWeightOptimizer, FeatureGroupWeightResult
    )
    from ..hdbscan_clustering.config.data_driven_config import (
        DataDrivenClusteringConfig, FeatureGroupWeightConfig
    )
    DATA_DRIVEN_AVAILABLE = True
except ImportError:
    DATA_DRIVEN_AVAILABLE = False
    tprint("⚠️ Data-driven optimization not available, will use hardcoded weights", "WARNING")

# Import weighted category PCA
try:
    from .weighted_category_pca import WeightedCategoryPCA, create_feature_categories_from_names
    WEIGHTED_PCA_AVAILABLE = True
except ImportError:
    WEIGHTED_PCA_AVAILABLE = False
    tprint("⚠️ WeightedCategoryPCA not available, will use standard PCA", "WARNING")

# Import CV enhancement strategies
try:
    from .cv_enhancement_strategies import (
        apply_cv_enhancement_strategies,
        RegimeDiscriminativeFeatures
    )
    CV_ENHANCEMENT_AVAILABLE = True
except ImportError:
    CV_ENHANCEMENT_AVAILABLE = False
    tprint("⚠️ CV enhancement strategies not available", "WARNING")

# Optional imports
try:
    import umap
except ImportError:
    umap = None

from .shared_utils import (
    prepare_market_features,
    FeatureConfig,
    FeaturePreparationResult,
    get_logger
)

@dataclass
class ClusteringContext:
    """Context for clustering operations."""
    original_features: np.ndarray
    market_data: pd.DataFrame
    memory_optimizer: Any = None
    original_feature_names: Optional[List[str]] = None
    feature_scores: Optional[Dict[str, float]] = None

    # Outputs
    optimized_features: Optional[np.ndarray] = None
    optimized_feature_names: Optional[List[str]] = None
    dropped_feature_names: Optional[List[str]] = None
    pca_loading_scores: Optional[Dict[str, float]] = None
    pre_pca_feature_names: Optional[List[str]] = None
    pre_pca_feature_count: Optional[int] = None

    # Data-driven optimization results
    data_driven_weights: Optional[Dict[str, float]] = None
    optimization_results: Optional[Dict[str, Any]] = None

    # Clustering results
    initial_assignments: Optional[np.ndarray] = None
    optimized_assignments: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None
    optimal_bic: Optional[float] = None
    k_metadata: Dict[str, Any] = field(default_factory=dict)
    tas_assignments: Optional[np.ndarray] = None
    nas_assignments: Optional[np.ndarray] = None
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)
    raw_assignments: Optional[np.ndarray] = None
    smoothed_assignments: Optional[np.ndarray] = None
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

class DataDrivenFeaturePreparationStep:
    """Step 1: Data-driven feature preparation and optimization."""

    def __init__(self, verbose: bool = True, enable_data_driven: bool = True):
        """Initialize the data-driven feature preparation step."""
        self.verbose = verbose
        self.enable_data_driven = enable_data_driven and DATA_DRIVEN_AVAILABLE
        self.logger = get_logger('DataDrivenFeaturePreparationStep')
        
        # Initialize data-driven optimizer if available
        if self.enable_data_driven:
            self.feature_weight_optimizer = DataDrivenFeatureWeightOptimizer(
                FeatureGroupWeightConfig()
            )
        
        # Initialize enhanced utilities
        try:
            self.hardware_manager = get_integrated_hardware_manager()
            self.vectorization_manager = UnifiedVectorizationManager(VectorizationConfig())
            self.data_utils = UnifiedDataUtils()
            tprint_debug("Enhanced utilities initialized for feature preparation")
        except Exception as e:
            tprint_warning(f"Failed to initialize enhanced utilities: {e}")
            self.hardware_manager = None
            self.vectorization_manager = None
            self.data_utils = None

    async def execute(self, context: ClusteringContext, config: Any) -> ClusteringContext:
        """Execute data-driven feature preparation step."""
        try:
            tprint("Step 1: Starting data-driven feature preparation and optimization...", "INFO")

            # Step 1a: Add regime-discriminative features to market data (BEFORE feature extraction)
            use_cv_enhancement = getattr(config, 'use_cv_enhancement', True)
            if use_cv_enhancement and CV_ENHANCEMENT_AVAILABLE:
                try:
                    tprint("⭐ Applying CV enhancement strategies to market data...", "INFO")
                    context.market_data = apply_cv_enhancement_strategies(
                        context.market_data,
                        add_regime_features=True
                    )
                except Exception as e:
                    tprint(f"⚠️ CV enhancement failed, continuing without it: {e}", "WARNING")

            # Step 1b: Use shared utilities for feature preparation
            feature_result = await self._prepare_features_using_shared_utils(
                context.market_data, config
            )

            # Step 1c: Inject prepared feature data into the context before optimization
            context = self._integrate_feature_result_into_context(context, feature_result)

            # Step 1d: Apply data-driven feature optimization
            context = await self._optimize_features_data_driven(context, config)

            tprint("Step 1: Data-driven feature preparation completed successfully", "SUCCESS")
            return context

        except Exception as e:
            tprint(f"Step 1: Data-driven feature preparation failed: {e}", "ERROR")
            raise ValueError(f"Data-driven feature preparation failed: {e}")

    def _integrate_feature_result_into_context(
        self,
        context: ClusteringContext,
        feature_result: FeaturePreparationResult
    ) -> ClusteringContext:
        """Populate the clustering context with shared utility feature outputs."""

        if feature_result is None:
            raise ValueError("Shared feature preparation returned no result")

        # Extract feature matrix
        if hasattr(feature_result, 'features') and feature_result.features is not None:
            features = feature_result.features
            feature_names = getattr(feature_result, 'feature_names', None)
            feature_scores = getattr(feature_result, 'feature_scores', None)
            dropped_features = getattr(feature_result, 'dropped_features', None)
            metadata = getattr(feature_result, 'metadata', {}) or {}
        elif hasattr(feature_result, 'features_array') and feature_result.features_array is not None:
            features = feature_result.features_array
            feature_df = getattr(feature_result, 'features_df', None)
            feature_names = list(feature_df.columns) if feature_df is not None else None
            metadata = getattr(feature_result, 'metadata', {}) or {}
            feature_scores = metadata.get('feature_scores') or metadata.get('scores')
            dropped_features = metadata.get('dropped_columns')
        else:
            features = np.asarray(feature_result)
            feature_names = None
            feature_scores = None
            metadata = {}
            dropped_features = None

        if features is None or getattr(features, 'size', 0) == 0:
            raise ValueError("Shared feature preparation produced empty features")

        # Derive feature names if missing
        n_features = features.shape[1] if hasattr(features, 'shape') and len(features.shape) >= 2 else 0
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Normalize feature scores
        if feature_scores is None:
            feature_scores = {}
        feature_scores = {
            str(name): float(score)
            for name, score in (feature_scores.items() if isinstance(feature_scores, dict) else [])
        }

        # Extract dropped feature names from metadata structures
        dropped_feature_names: List[str] = []
        if isinstance(dropped_features, list):
            dropped_feature_names.extend(str(name) for name in dropped_features)
        elif isinstance(dropped_features, dict):
            for names in dropped_features.values():
                if isinstance(names, (list, tuple, set)):
                    dropped_feature_names.extend(str(name) for name in names)

        stage_metadata = metadata.get('stage_metadata') if isinstance(metadata, dict) else None
        if isinstance(stage_metadata, dict):
            operations = stage_metadata.get('operations', [])
            for operation in operations:
                if isinstance(operation, dict):
                    removed = operation.get('removed_columns') or operation.get('removed_features')
                    if isinstance(removed, list):
                        dropped_feature_names.extend(str(name) for name in removed)

        # Deduplicate dropped feature names while preserving order
        if dropped_feature_names:
            seen = set()
            deduped = []
            for name in dropped_feature_names:
                if name not in seen:
                    deduped.append(name)
                    seen.add(name)
            dropped_feature_names = deduped

        # Update context attributes with shared utility outputs
        context.original_features = features
        context.original_feature_names = list(feature_names)
        context.feature_scores = feature_scores
        context.dropped_feature_names = dropped_feature_names
        context.pre_pca_feature_names = list(feature_names)
        context.pre_pca_feature_count = len(feature_names)

        # Store metadata for downstream reporting without overwriting existing summary
        context.summary = context.summary or {}
        context.summary.setdefault('feature_preparation', {})
        context.summary['feature_preparation'].update({
            'metadata': metadata,
            'feature_count': len(feature_names),
            'dropped_features': dropped_feature_names,
        })

        return context

    async def _prepare_features_using_shared_utils(
        self,
        market_data: pd.DataFrame,
        config: Any
    ) -> FeaturePreparationResult:
        """Prepare features using shared utilities."""
        try:
            # Use shared feature configuration
            feature_config = FeatureConfig(
                feature_categories=getattr(config, 'feature_categories', [
                    'regime_volatility',
                    'regime_volume',
                    'regime_structural_trend',
                    'regime_statistical'
                ]),
                use_standardized_features=getattr(config, 'use_standardized_features', True),
                drop_highly_correlated=True
            )

            # Prepare features using shared utilities
            feature_result = prepare_market_features(
                market_data=market_data,
                feature_config=feature_config,
                return_metadata=True
            )

            features = feature_result.features_array
            tprint(f"Shared utilities prepared {features.shape[1]} features", "SUCCESS")

            return feature_result

        except Exception as e:
            tprint(f"Shared feature preparation failed: {e}", "ERROR")
            raise

    async def _optimize_features_data_driven(self, context: ClusteringContext, config: Any) -> ClusteringContext:
        """Optimize features using data-driven dimensionality reduction."""
        try:
            tprint("Starting data-driven feature optimization...", "INFO")
            tprint(f"🔍 DEBUG: Original features shape: {context.original_features.shape}", "INFO")

            # Step 1: Standardize features with updated feature tracking
            tprint("Step 1: Standardizing features using RobustScaler for financial data...", "INFO")
            scaler = RobustScaler()

            feature_names = context.original_feature_names or [
                f"feature_{i}" for i in range(context.original_features.shape[1])
            ]
            context.original_feature_names = list(feature_names)
            context.pre_pca_feature_names = list(feature_names)
            context.pre_pca_feature_count = len(feature_names)

            features_scaled = scaler.fit_transform(context.original_features)
            tprint(f"Feature standardization completed: {context.original_features.shape}", "SUCCESS")
            tprint(f"🔍 MEMORY: Scaled features created - {features_scaled.nbytes / 1024 / 1024:.2f} MB", "INFO")

            # Step 2: Data-driven feature weight optimization
            if self.enable_data_driven:
                try:
                    tprint("⭐ Starting data-driven feature weight optimization...", "INFO")
                    
                    # Create a simple clustering function for optimization
                    def simple_clustering_func(features):
                        from sklearn.cluster import KMeans
                        n_clusters = min(5, features.shape[0] // 10)
                        if n_clusters < 2:
                            n_clusters = 2
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        return kmeans.fit_predict(features)
                    
                    # Optimize feature weights
                    weight_result = self.feature_weight_optimizer.optimize_weights(
                        features=features_scaled,
                        feature_names=feature_names,
                        market_data=context.market_data,
                        clustering_func=simple_clustering_func
                    )
                    
                    # Store optimization results
                    context.data_driven_weights = weight_result.optimal_weights
                    context.optimization_results = {
                        'feature_weights': weight_result.__dict__
                    }
                    
                    tprint(f"✅ Data-driven weights: {weight_result.optimal_weights}", "SUCCESS")
                    
                    # Apply optimized weights
                    features_scaled = self._apply_data_driven_weights(
                        features_scaled, feature_names, weight_result.optimal_weights
                    )
                    
                except Exception as e:
                    tprint(f"⚠️ Data-driven optimization failed: {e}, falling back to hardcoded weights", "WARNING")
                    # Fall back to hardcoded weights
                    features_scaled = self._apply_hardcoded_weights(features_scaled, feature_names)
            else:
                # Use hardcoded weights
                features_scaled = self._apply_hardcoded_weights(features_scaled, feature_names)

            # Step 3: Apply dimensionality reduction
            if context.original_features.shape[1] < 2:
                tprint_warning("⚠️ Fewer than two features available after pruning - skipping PCA")
                features_final = self._validate_feature_quality_minimal(features_scaled, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = list(feature_names)
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {name: 1.0 for name in feature_names}
                if context.feature_scores:
                    context.feature_scores = {
                        name: float(context.feature_scores.get(name, 0.0)) for name in feature_names
                    }
                return context

            # Try Weighted Category PCA first (ENHANCED APPROACH)
            use_weighted_pca = getattr(config, 'use_weighted_category_pca', True)
            if WEIGHTED_PCA_AVAILABLE and use_weighted_pca and context.original_features.shape[1] >= 4:
                tprint("⭐ Attempting Weighted Category PCA (ENHANCED APPROACH)...", "INFO")
                try:
                    # Auto-detect categories from feature names
                    categories = create_feature_categories_from_names(feature_names)

                    if categories:
                        # Create and fit transformer
                        pca_transformer = WeightedCategoryPCA(categories_config=categories)
                        features_pca = pca_transformer.fit_transform(features_scaled, feature_names)

                        # Get transformed feature names and summary
                        transformed_names = pca_transformer.get_feature_names_out()
                        component_summary = pca_transformer.get_component_summary()

                        # Validate features
                        features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)

                        # Update context
                        context.optimized_features = features_final
                        context.optimized_feature_names = transformed_names
                        context.dropped_feature_names = context.dropped_feature_names or []

                        # Create PCA loading scores from variance explained
                        pca_loading_scores = {}
                        for cat_name, cat_info in component_summary.items():
                            cat_weight = categories[cat_name].weight
                            for i, var_explained in enumerate(cat_info['explained_variance_ratio']):
                                comp_name = f"{cat_name}_pc{i+1}"
                                # Score = variance explained * category weight
                                pca_loading_scores[comp_name] = float(var_explained * cat_weight)

                        context.pca_loading_scores = pca_loading_scores
                        if context.feature_scores:
                            context.feature_scores = pca_loading_scores

                        tprint(f"✅ Weighted Category PCA Success: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")

                        # Save transformer for later use (test-time transformation)
                        try:
                            import os
                            os.makedirs('models/pca', exist_ok=True)
                            pca_transformer.save('models/pca/weighted_category_pca.pkl')
                        except Exception as save_err:
                            tprint(f"⚠️ Could not save PCA transformer: {save_err}", "WARNING")

                        self._safe_memory_cleanup([features_scaled, features_pca])
                        return context
                    else:
                        tprint("⚠️ No feature categories detected, falling back to standard PCA", "WARNING")
                except Exception as pca_err:
                    tprint(f"⚠️ Weighted Category PCA failed: {pca_err}, falling back to standard PCA", "WARNING")

            # Try UMAP reduction as an alternative to PCA
            umap_features = self._try_umap_reduction(features_scaled, target_features=20)
            if umap_features is not None:
                tprint("Using UMAP reduction instead of PCA", "INFO")
                features_final = self._validate_feature_quality_minimal(umap_features, context.market_data)

                # Create meaningful UMAP feature names
                umap_feature_names = [f"UMAP_dim{i+1}" for i in range(features_final.shape[1])]

                context.optimized_features = features_final
                context.optimized_feature_names = umap_feature_names
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {umap_feature_names[i]: 1.0 for i in range(features_final.shape[1])}
                if context.feature_scores:
                    context.feature_scores = {umap_feature_names[i]: 1.0 for i in range(features_final.shape[1])}

                tprint(f"UMAP feature optimization: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")

                self._safe_memory_cleanup([features_scaled, umap_features])
                return context

            # Fallback to standard PCA with data-driven or hardcoded weights
            tprint("🔧 Using standard PCA for dimensionality reduction", "INFO")
            n_samples, n_features = features_scaled.shape
            n_components_total = min(20, n_features - 1)
            
            pca = PCA(n_components=n_components_total, whiten=True, random_state=42)
            features_pca = pca.fit_transform(features_scaled)
            
            # Create PCA feature names
            pca_feature_names = [f"PC{i+1}_var{pca.explained_variance_ratio_[i]:.3f}" for i in range(features_pca.shape[1])]
            
            # Validate features
            features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)
            
            # Update context
            context.optimized_features = features_final
            context.optimized_feature_names = pca_feature_names
            context.dropped_feature_names = context.dropped_feature_names or []
            context.pca_loading_scores = {pca_feature_names[i]: float(pca.explained_variance_ratio_[i]) for i in range(len(pca_feature_names))}
            
            if context.feature_scores:
                context.feature_scores = {pca_feature_names[i]: float(pca.explained_variance_ratio_[i]) for i in range(len(pca_feature_names))}

            tprint(f"📈 Standard PCA: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")

            self._safe_memory_cleanup([features_scaled])

        except Exception as e:
            tprint(f"Data-driven feature optimization failed: {e}", "ERROR")
            raise ValueError(f"Data-driven feature optimization failed: {e}")

        return context

    def _apply_data_driven_weights(self, 
                                 features: np.ndarray, 
                                 feature_names: List[str], 
                                 weights: Dict[str, float]) -> np.ndarray:
        """Apply data-driven feature group weights."""
        try:
            weighted_features = features.copy()
            
            # Categorize features
            feature_groups = self._categorize_features(feature_names)
            
            # Apply weights to each group
            for group, group_features in feature_groups.items():
                if group in weights:
                    # Find indices of features in this group
                    feature_indices = [i for i, name in enumerate(feature_names) if name in group_features]
                    
                    # Apply weight (sqrt because we're scaling variance)
                    weight = np.sqrt(weights[group])
                    weighted_features[:, feature_indices] *= weight
                    
                    tprint(f"Applied weight {weights[group]:.3f} to {group} group ({len(feature_indices)} features)", "DEBUG")
            
            return weighted_features
            
        except Exception as e:
            tprint(f"Data-driven weight application failed: {e}", "WARNING")
            return features

    def _apply_hardcoded_weights(self, 
                               features: np.ndarray, 
                               feature_names: List[str]) -> np.ndarray:
        """Apply hardcoded feature group weights (fallback)."""
        try:
            # Use the original hardcoded weights as fallback
            w_returns, w_vol, w_volume = 0.50, 0.30, 0.20
            
            # Categorize features
            feature_groups = self._categorize_features(feature_names)
            
            # Apply hardcoded weights
            weighted_features = features.copy()
            
            if 'returns' in feature_groups:
                feature_indices = [i for i, name in enumerate(feature_names) if name in feature_groups['returns']]
                weighted_features[:, feature_indices] *= np.sqrt(w_returns)
                
            if 'volatility' in feature_groups:
                feature_indices = [i for i, name in enumerate(feature_names) if name in feature_groups['volatility']]
                weighted_features[:, feature_indices] *= np.sqrt(w_vol)
                
            if 'volume' in feature_groups:
                feature_indices = [i for i, name in enumerate(feature_names) if name in feature_groups['volume']]
                weighted_features[:, feature_indices] *= np.sqrt(w_volume)
            
            tprint(f"Applied hardcoded weights: returns={w_returns}, vol={w_vol}, volume={w_volume}", "INFO")
            return weighted_features
            
        except Exception as e:
            tprint(f"Hardcoded weight application failed: {e}", "WARNING")
            return features

    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features into groups based on naming patterns."""
        groups = {'returns': [], 'volatility': [], 'volume': [], 'other': []}
        
        for name in feature_names:
            name_lower = name.lower()
            
            # Returns group
            if any(term in name_lower for term in ['return', 'log_return', 'close_return', 'pct_change']):
                groups['returns'].append(name)
            # Volatility group
            elif any(term in name_lower for term in ['volatility', 'vol_', 'atr', 'std', 'boll', 'bb']):
                groups['volatility'].append(name)
            # Volume group
            elif any(term in name_lower for term in ['volume', 'vwap', 'obv', 'accumulation', 'distribution']):
                groups['volume'].append(name)
            # Other group
            else:
                groups['other'].append(name)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups

    def _try_umap_reduction(self, features: np.ndarray, target_features: int = 20) -> Optional[np.ndarray]:
        """Try UMAP reduction as an alternative to PCA."""
        try:
            if umap is None or not hasattr(umap, 'UMAP'):
                return None

            reducer = umap.UMAP(
                n_components=target_features,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
            return reducer.fit_transform(features)
        except ImportError:
            tprint("UMAP not available, falling back to PCA", "INFO")
            return None
        except Exception as e:
            tprint(f"UMAP reduction failed: {e}, falling back to PCA", "WARNING")
            return None

    def _validate_feature_quality_minimal(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Validate feature quality with minimal checks."""
        try:
            # Basic validation
            if features.shape[0] == 0:
                raise ValueError("No samples in features")
            if features.shape[1] == 0:
                raise ValueError("No features available")

            # Check for NaN values
            if np.any(np.isnan(features)):
                tprint_warning("⚠️ NaN values detected in features, filling with zeros")
                features = np.nan_to_num(features, nan=0.0)

            # Check for infinite values
            if np.any(np.isinf(features)):
                tprint_warning("⚠️ Infinite values detected in features, clipping")
                features = np.clip(features, -1e6, 1e6)

            return features

        except Exception as e:
            tprint(f"Feature validation failed: {e}", "ERROR")
            raise

    def _safe_memory_cleanup(self, arrays: List[np.ndarray]) -> None:
        """Safely clean up memory by deleting arrays."""
        try:
            for arr in arrays:
                if arr is not None:
                    del arr
        except Exception as e:
            tprint(f"Memory cleanup warning: {e}", "WARNING")