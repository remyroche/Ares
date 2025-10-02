"""
Step 1: Feature Preparation for NAS-TAS Clustering.

This module handles feature selection, dimensionality reduction, and regime-specific
feature integration for the clustering process.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from ..shared_utils import (
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


class FeaturePreparationStep:
    """Step 1: Feature preparation and optimization."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the feature preparation step."""
        self.verbose = verbose
        self.logger = get_logger('FeaturePreparationStep')
        
    async def execute(self, context: ClusteringContext, config: Any) -> ClusteringContext:
        """Execute feature preparation step."""
        try:
            tprint("Step 1: Starting feature preparation and optimization...", "INFO")
            
            # Use shared utilities for feature preparation
            feature_result = await self._prepare_features_using_shared_utils(
                context.market_data, config
            )
            
            # Apply regime-specific feature optimization
            context = await self._optimize_features(context, config)
            
            tprint("Step 1: Feature preparation completed successfully", "SUCCESS")
            return context
            
        except Exception as e:
            tprint(f"Step 1: Feature preparation failed: {e}", "ERROR")
            raise ValueError(f"Feature preparation failed: {e}")
    
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
                config=feature_config
            )

            # Handle both return types: FeaturePreparationResult or numpy array
            if hasattr(feature_result, 'features'):
                # It's a FeaturePreparationResult
                features = feature_result.features
                tprint(f"Shared utilities prepared {features.shape[1]} features", "SUCCESS")
            else:
                # It's a numpy array
                features = feature_result
                tprint(f"Shared utilities prepared {features.shape[1]} features", "SUCCESS")

            # Ensure we return a FeaturePreparationResult-like object
            if hasattr(feature_result, 'features'):
                return feature_result
            else:
                # Create a FeaturePreparationResult-like object for consistency
                return FeaturePreparationResult(
                    features=features,
                    feature_names=[f'feature_{i}' for i in range(features.shape[1])],
                    feature_scores={},
                    dropped_features=[],
                    preparation_time=0.0,
                    metadata={'prepared_directly': True, 'total_features': features.shape[1]}
                )
            
        except Exception as e:
            tprint(f"Shared feature preparation failed: {e}", "ERROR")
            raise
    
    async def _optimize_features(self, context: ClusteringContext, config: Any) -> ClusteringContext:
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

            if context.original_features.shape[1] < 2:
                tprint_warning("⚠️ Fewer than two features available after pruning - skipping PCA")
                tprint(f"🔍 DEBUG: Insufficient features for PCA - only {context.original_features.shape[1]} features available", "WARNING")
                features_final = self._validate_feature_quality_minimal(features_scaled, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = list(feature_names)
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {name: 1.0 for name in feature_names}
                if context.feature_scores:
                    context.feature_scores = {
                        name: float(context.feature_scores.get(name, 0.0)) for name in feature_names
                    }

                tprint(
                    f"Data-driven feature optimization (PCA skipped): {context.original_features.shape} -> {features_final.shape}",
                    "SUCCESS",
                )

                self._safe_memory_cleanup([features_scaled])
                return context

            # Try UMAP reduction as an alternative to PCA
            umap_features = self._try_umap_reduction(features_scaled, target_features=20)
            if umap_features is not None:
                tprint("Using UMAP reduction instead of PCA", "INFO")
                features_final = self._validate_feature_quality_minimal(umap_features, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = [f"umap_{i}" for i in range(features_final.shape[1])]
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {f"umap_{i}": 1.0 for i in range(features_final.shape[1])}
                if context.feature_scores:
                    context.feature_scores = {f"umap_{i}": 1.0 for i in range(features_final.shape[1])}
                
                tprint(f"UMAP feature optimization: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")
                
                self._safe_memory_cleanup([features_scaled, umap_features])
                return context

            # Fallback to PCA
            tprint("Using PCA for dimensionality reduction", "INFO")
            pca = PCA(n_components=min(20, features_scaled.shape[1] - 1))
            features_pca = pca.fit_transform(features_scaled)
            
            features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)
            context.optimized_features = features_final
            context.optimized_feature_names = [f"pca_{i}" for i in range(features_final.shape[1])]
            context.dropped_feature_names = context.dropped_feature_names or []
            context.pca_loading_scores = {f"pca_{i}": float(pca.explained_variance_ratio_[i]) for i in range(features_final.shape[1])}
            if context.feature_scores:
                context.feature_scores = {f"pca_{i}": float(pca.explained_variance_ratio_[i]) for i in range(features_final.shape[1])}
            
            tprint(f"PCA feature optimization: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")
            
            self._safe_memory_cleanup([features_scaled, features_pca])

        except Exception as e:
            tprint(f"Feature optimization failed: {e}", "ERROR")
            raise ValueError(f"Feature optimization failed: {e}")
        
        return context

    def _try_umap_reduction(self, features: np.ndarray, target_features: int = 20) -> Optional[np.ndarray]:
        """Try UMAP reduction as an alternative to PCA."""
        try:
            import umap
            if not hasattr(umap, 'UMAP'):
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