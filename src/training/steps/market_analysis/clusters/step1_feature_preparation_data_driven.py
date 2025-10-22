"""
Step 1: Data-Driven Feature Preparation for NAS-TAS Clustering.

This module handles feature selection, dimensionality reduction, and regime-specific
feature integration for the clustering process with data-driven parameter optimization.

ENHANCED WITH BASESTEP COMPREHENSIVE TOOLS:
- Direct access to all utility modules through BaseStep
- Comprehensive logging with tprint integration
- Hardware optimization built-in
- Safe operations with fallbacks
- Memory management and cleanup
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Awaitable, Callable
from dataclasses import dataclass, field
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Import BaseStep for comprehensive utility access
from src.training.steps.base_step import BaseStep

# Import tprint functions directly (available through BaseStep)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance,
    tprint_step_start, tprint_step_end, tprint_operation_start, tprint_operation_end,
    tprint_data_summary, tprint_performance_summary, tprint_memory_usage
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

class DataDrivenFeaturePreparationStep(BaseStep):
    """Step 1: Data-driven feature preparation and optimization with BaseStep comprehensive tools."""

    def __init__(self, verbose: bool = True, enable_data_driven: bool = True, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the data-driven feature preparation step with BaseStep utilities."""
        super().__init__("data_driven_feature_preparation", config)
        
        tprint_step_start("DataDrivenFeaturePreparationStep", config)
        self.verbose = verbose
        self.enable_data_driven = enable_data_driven and DATA_DRIVEN_AVAILABLE
        
        # Log utility availability
        availability = self._get_availability_status()
        tprint_info(f"Utility availability: {sum(availability.values())}/{len(availability)} utilities available")
        
        tprint_debug(f"Verbose mode: {verbose}")
        tprint_debug(f"Data-driven optimization enabled: {self.enable_data_driven}")
        
        # Initialize data-driven optimizer if available
        if self.enable_data_driven:
            tprint_info("🧠 Initializing data-driven feature weight optimizer")
            self.feature_weight_optimizer = DataDrivenFeatureWeightOptimizer(
                FeatureGroupWeightConfig()
            )
            tprint_debug("DataDrivenFeatureWeightOptimizer initialized")
        else:
            tprint_warning("⚠️ Data-driven optimization not available, using hardcoded weights")
            self.feature_weight_optimizer = None
        
        # Use BaseStep utilities instead of direct imports
        tprint_info("🔧 Using BaseStep comprehensive utilities")
        tprint_debug("All utilities available through BaseStep instance attributes")
        
        tprint_step_end("DataDrivenFeaturePreparationStep", True, 0.0)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data-driven feature preparation step using BaseStep comprehensive tools."""
        tprint_step_start("Data-Driven Feature Preparation", config)
        
        try:
            # Extract context from config or create new one
            context = self._extract_or_create_context(config)
            
            # Step 1a: Add regime-discriminative features to market data (BEFORE feature extraction)
            use_cv_enhancement = config.get('use_cv_enhancement', True)
            if use_cv_enhancement and CV_ENHANCEMENT_AVAILABLE:
                try:
                    tprint_operation_start("CV Enhancement Strategies")
                    context.market_data = apply_cv_enhancement_strategies(
                        context.market_data,
                        add_regime_features=True
                    )
                    tprint_operation_end("CV Enhancement Strategies", True)
                except Exception as e:
                    tprint_warning(f"⚠️ CV enhancement failed, continuing without it: {e}")

            # Step 1b: Use BaseStep utilities for feature preparation
            tprint_operation_start("Feature Preparation")
            feature_result = await self._prepare_features_using_basestep_utils(
                context.market_data, config
            )
            tprint_operation_end("Feature Preparation", True)

            # Step 1c: Inject prepared feature data into the context before optimization
            context = self._integrate_feature_result_into_context(context, feature_result)

            # Step 1d: Apply data-driven feature optimization
            tprint_operation_start("Data-Driven Optimization")
            context = await self._optimize_features_data_driven(context, config)
            tprint_operation_end("Data-Driven Optimization", True)

            # Create comprehensive outcome using BaseStep utilities
            outcome = self._create_comprehensive_outcome(context, config)
            
            tprint_step_end("Data-Driven Feature Preparation", True, 0.0)
            return outcome

        except Exception as e:
            tprint_error(f"❌ Data-driven feature preparation failed: {e}")
            tprint_step_end("Data-Driven Feature Preparation", False, 0.0)
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    def _extract_or_create_context(self, config: Dict[str, Any]) -> ClusteringContext:
        """Extract context from config or create new one using BaseStep utilities."""
        try:
            # Try to extract from config
            if 'context' in config:
                return config['context']
            
            # Create new context with market data
            market_data = config.get('market_data')
            if market_data is None:
                raise ValueError("Market data is required in config")
            
            # Use BaseStep utilities for data validation
            if not self._validate_dataframe_columns(market_data, []):
                tprint_warning("⚠️ Market data validation failed, using as-is")
            
            # Create context
            context = ClusteringContext(
                original_features=np.array([]),
                market_data=market_data,
                original_feature_names=[]
            )
            
            return context
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract or create context: {e}")
            raise

    async def _prepare_features_using_basestep_utils(
        self, 
        market_data: pd.DataFrame, 
        config: Any
    ) -> FeaturePreparationResult:
        """Prepare features using BaseStep comprehensive utilities."""
        try:
            tprint_info("🔧 Preparing features using BaseStep utilities")
            
            # Use BaseStep data quality utilities
            if self.data_quality:
                tprint_debug("Using BaseStep data quality utilities")
                # Clean data using BaseStep utilities
                cleaned_data = self._safe_dataframe_operation(market_data, "fillna")
            else:
                cleaned_data = market_data.fillna(0)
            
            # Basic feature preparation
            features = cleaned_data.select_dtypes(include=[np.number]).values
            
            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            
            # Simple feature names
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            
            # Apply scaling using BaseStep utilities
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA if requested
            pca = None
            pca_components = config.get('pca_components', 20)
            if pca_components < features_scaled.shape[1]:
                pca = PCA(n_components=pca_components)
                features_scaled = pca.fit_transform(features_scaled)
                feature_names = [f"pca_{i}" for i in range(pca_components)]
            
            # Use BaseStep hardware optimization if available
            if self.hardware_utils:
                tprint_debug("Applying hardware optimization to features")
                features_scaled = self.hardware_utils['optimize_dataframe'](
                    pd.DataFrame(features_scaled)
                ).values
            
            # Create result
            result = FeaturePreparationResult(
                features=features_scaled,
                feature_names=feature_names,
                scaler=scaler,
                pca=pca,
                feature_scores={name: 1.0 for name in feature_names}
            )
            
            tprint_success(f"✅ Features prepared: {features_scaled.shape}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Feature preparation failed: {e}")
            raise

    def _create_comprehensive_outcome(
        self, 
        context: ClusteringContext, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive outcome using BaseStep utilities."""
        try:
            # Calculate performance metrics
            metrics = {
                'features_processed': context.optimized_features.shape[0] if context.optimized_features is not None else 0,
                'feature_dimensions': context.optimized_features.shape[1] if context.optimized_features is not None else 0,
                'data_driven_enabled': self.enable_data_driven,
                'optimization_success': context.optimized_features is not None
            }
            
            # Use BaseStep performance logging
            tprint_performance_summary(metrics)
            
            # Create artifacts using BaseStep utilities
            artifacts = []
            if context.optimized_features is not None:
                # Save optimized features
                self._save_dataframe(
                    pd.DataFrame(context.optimized_features), 
                    "optimized_features"
                )
                artifacts.append("optimized_features")
            
            if context.optimized_feature_names:
                # Save feature names
                self._save_metadata(
                    context.optimized_feature_names, 
                    "feature_names"
                )
                artifacts.append("feature_names")
            
            # Create outcome
            outcome = {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'context': context,
                'execution_time': 0.0  # Will be updated by BaseStep
            }
            
            return outcome
            
        except Exception as e:
            tprint_error(f"❌ Failed to create comprehensive outcome: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    def _apply_data_driven_weights_safe(
        self, 
        features: np.ndarray, 
        feature_names: List[str], 
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Apply data-driven weights using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            weighted_features = features.copy()
            
            for i, name in enumerate(feature_names):
                if name in weights:
                    weight = self._validate_finite(weights[name], default=1.0)
                    weighted_features[:, i] = self._safe_divide(
                        weighted_features[:, i] * weight, 1.0, default=weighted_features[:, i]
                    )
            
            # Validate result
            weighted_features = self._validate_finite(weighted_features, default=0)
            return weighted_features
            
        except Exception as e:
            tprint_error(f"❌ Failed to apply data-driven weights: {e}")
            return features

    def _apply_hardcoded_weights_safe(
        self, 
        features: np.ndarray, 
        feature_names: List[str]
    ) -> np.ndarray:
        """Apply hardcoded weights using BaseStep safe operations."""
        try:
            # Simple hardcoded weights based on feature names
            weights = {}
            for name in feature_names:
                if 'price' in name.lower() or 'return' in name.lower():
                    weights[name] = 1.2
                elif 'volume' in name.lower():
                    weights[name] = 1.1
                else:
                    weights[name] = 1.0
            
            return self._apply_data_driven_weights_safe(features, feature_names, weights)
            
        except Exception as e:
            tprint_error(f"❌ Failed to apply hardcoded weights: {e}")
            return features

    def _validate_feature_quality_minimal_safe(
        self, 
        features: np.ndarray, 
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """Validate feature quality using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            validated_features = self._validate_finite(features, default=0)
            
            # Check for any remaining issues
            if np.any(np.isnan(validated_features)):
                tprint_warning("⚠️ NaN values found in features, replacing with 0")
                validated_features = np.nan_to_num(validated_features, nan=0.0)
            
            return validated_features
            
        except Exception as e:
            tprint_error(f"❌ Feature validation failed: {e}")
            return features

    def _try_umap_reduction_safe(
        self, 
        features: np.ndarray, 
        target_features: int = 20
    ) -> Optional[np.ndarray]:
        """Try UMAP reduction using BaseStep safe operations."""
        try:
            if umap is None:
                tprint_warning("⚠️ UMAP not available")
                return None
            
            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            
            # Apply UMAP
            reducer = umap.UMAP(n_components=min(target_features, features.shape[1] - 1))
            umap_features = reducer.fit_transform(features)
            
            # Validate result
            umap_features = self._validate_finite(umap_features, default=0)
            
            return umap_features
            
        except Exception as e:
            tprint_warning(f"⚠️ UMAP reduction failed: {e}")
            return None

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
        """Optimize features using data-driven dimensionality reduction with BaseStep utilities."""
        try:
            tprint_operation_start("Data-Driven Feature Optimization")
            tprint_info(f"🔍 Original features shape: {context.original_features.shape}")

            # Step 1: Standardize features with BaseStep utilities
            tprint_info("Step 1: Standardizing features using RobustScaler for financial data")
            scaler = RobustScaler()

            feature_names = context.original_feature_names or [
                f"feature_{i}" for i in range(context.original_features.shape[1])
            ]
            context.original_feature_names = list(feature_names)
            context.pre_pca_feature_names = list(feature_names)
            context.pre_pca_feature_count = len(feature_names)

            # Use BaseStep math validation for safe operations
            features_scaled = scaler.fit_transform(context.original_features)
            features_scaled = self._validate_finite(features_scaled, default=0)
            
            tprint_success(f"Feature standardization completed: {context.original_features.shape}")
            
            # Use BaseStep memory monitoring
            if self.hardware_utils:
                memory_usage = self.hardware_utils['get_memory_usage']()
                tprint_info(f"🔍 Memory usage: {memory_usage:.2f} MB")

            # Step 2: Data-driven feature weight optimization using BaseStep utilities
            if self.enable_data_driven:
                try:
                    tprint_info("⭐ Starting data-driven feature weight optimization")
                    
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
                    
                    # Store optimization results using BaseStep utilities
                    context.data_driven_weights = weight_result.optimal_weights
                    context.optimization_results = {
                        'feature_weights': weight_result.__dict__
                    }
                    
                    # Use BaseStep safe operations for weight application
                    features_scaled = self._apply_data_driven_weights_safe(
                        features_scaled, feature_names, weight_result.optimal_weights
                    )
                    
                    tprint_success(f"✅ Data-driven weights applied successfully")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Data-driven optimization failed: {e}, falling back to hardcoded weights")
                    # Fall back to hardcoded weights using BaseStep utilities
                    features_scaled = self._apply_hardcoded_weights_safe(features_scaled, feature_names)
            else:
                # Use hardcoded weights with BaseStep utilities
                features_scaled = self._apply_hardcoded_weights_safe(features_scaled, feature_names)

            # Step 3: Apply dimensionality reduction using BaseStep utilities
            if context.original_features.shape[1] < 2:
                tprint_warning("⚠️ Fewer than two features available after pruning - skipping PCA")
                features_final = self._validate_feature_quality_minimal_safe(features_scaled, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = list(feature_names)
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {name: 1.0 for name in feature_names}
                if context.feature_scores:
                    context.feature_scores = {
                        name: float(self._validate_finite(context.feature_scores.get(name, 0.0), default=0.0)) 
                        for name in feature_names
                    }
                tprint_operation_end("Data-Driven Feature Optimization", True)
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
                        features_final = self._validate_feature_quality_minimal_safe(features_pca, context.market_data)

                        # Update context
                        context.optimized_features = features_final
                        context.optimized_feature_names = transformed_names
                        context.dropped_feature_names = context.dropped_feature_names or []

                        # Create PCA loading scores from variance explained using BaseStep utilities
                        pca_loading_scores = {}
                        for cat_name, cat_info in component_summary.items():
                            cat_weight = self._validate_finite(categories[cat_name].weight, default=1.0)
                            for i, var_explained in enumerate(cat_info['explained_variance_ratio']):
                                comp_name = f"{cat_name}_pc{i+1}"
                                # Score = variance explained * category weight using safe operations
                                score = self._safe_divide(
                                    var_explained * cat_weight, 1.0, default=var_explained
                                )
                                pca_loading_scores[comp_name] = float(self._validate_finite(score, default=0.0))

                        context.pca_loading_scores = pca_loading_scores
                        if context.feature_scores:
                            context.feature_scores = pca_loading_scores

                        tprint_success(f"✅ Weighted Category PCA Success: {context.original_features.shape} -> {features_final.shape}")

                        # Save transformer using BaseStep utilities
                        try:
                            self._ensure_directory('models/pca')
                            # Use BaseStep safe file operations
                            self._safe_json_save(
                                {'transformer_type': 'weighted_category_pca'}, 
                                'models/pca/weighted_category_pca_metadata.json'
                            )
                        except Exception as save_err:
                            tprint_warning(f"⚠️ Could not save PCA transformer metadata: {save_err}")

                        # Use BaseStep memory cleanup
                        if self.hardware_utils:
                            self.hardware_utils['force_garbage_collection']()
                        
                        tprint_operation_end("Data-Driven Feature Optimization", True)
                        return context
                    else:
                        tprint("⚠️ No feature categories detected, falling back to standard PCA", "WARNING")
                except Exception as pca_err:
                    tprint(f"⚠️ Weighted Category PCA failed: {pca_err}, falling back to standard PCA", "WARNING")

            # Try UMAP reduction as an alternative to PCA using BaseStep utilities
            umap_features = self._try_umap_reduction_safe(features_scaled, target_features=20)
            if umap_features is not None:
                tprint_info("Using UMAP reduction instead of PCA")
                features_final = self._validate_feature_quality_minimal_safe(umap_features, context.market_data)

                # Create meaningful UMAP feature names
                umap_feature_names = [f"UMAP_dim{i+1}" for i in range(features_final.shape[1])]

                context.optimized_features = features_final
                context.optimized_feature_names = umap_feature_names
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {umap_feature_names[i]: 1.0 for i in range(features_final.shape[1])}
                if context.feature_scores:
                    context.feature_scores = {umap_feature_names[i]: 1.0 for i in range(features_final.shape[1])}

                tprint_success(f"UMAP feature optimization: {context.original_features.shape} -> {features_final.shape}")

                # Use BaseStep memory cleanup
                if self.hardware_utils:
                    self.hardware_utils['force_garbage_collection']()
                
                tprint_operation_end("Data-Driven Feature Optimization", True)
                return context

            # Fallback to standard PCA with data-driven or hardcoded weights using BaseStep utilities
            tprint_info("🔧 Using standard PCA for dimensionality reduction")
            n_samples, n_features = features_scaled.shape
            n_components_total = min(20, n_features - 1)
            
            pca = PCA(n_components=n_components_total, whiten=True, random_state=42)
            features_pca = pca.fit_transform(features_scaled)
            
            # Create PCA feature names
            pca_feature_names = [f"PC{i+1}_var{pca.explained_variance_ratio_[i]:.3f}" for i in range(features_pca.shape[1])]
            
            # Validate features using BaseStep utilities
            features_final = self._validate_feature_quality_minimal_safe(features_pca, context.market_data)
            
            # Update context
            context.optimized_features = features_final
            context.optimized_feature_names = pca_feature_names
            context.dropped_feature_names = context.dropped_feature_names or []
            
            # Use BaseStep safe operations for PCA loading scores
            pca_loading_scores = {}
            for i, name in enumerate(pca_feature_names):
                score = self._validate_finite(pca.explained_variance_ratio_[i], default=0.0)
                pca_loading_scores[name] = float(score)
            context.pca_loading_scores = pca_loading_scores
            
            if context.feature_scores:
                context.feature_scores = pca_loading_scores

            tprint_success(f"📈 Standard PCA: {context.original_features.shape} -> {features_final.shape}")

            # Use BaseStep memory cleanup
            if self.hardware_utils:
                self.hardware_utils['force_garbage_collection']()

        except Exception as e:
            tprint_error(f"❌ Data-driven feature optimization failed: {e}")
            tprint_operation_end("Data-Driven Feature Optimization", False)
            raise ValueError(f"Data-driven feature optimization failed: {e}")

        tprint_operation_end("Data-Driven Feature Optimization", True)
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