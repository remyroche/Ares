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

            # Step 1a: Add regime-discriminative features to market data (BEFORE feature extraction)
            use_cv_enhancement = getattr(config, 'use_cv_enhancement', True)  # Default: enabled
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

            # Step 1d: Apply regime-specific feature optimization (PCA, etc.)
            context = await self._optimize_features(context, config)

            tprint("Step 1: Feature preparation completed successfully", "SUCCESS")
            return context

        except Exception as e:
            tprint(f"Step 1: Feature preparation failed: {e}", "ERROR")
            raise ValueError(f"Feature preparation failed: {e}")

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

            # Try Weighted Category PCA first (ENHANCED APPROACH)
            use_weighted_pca = getattr(config, 'use_weighted_category_pca', True)  # Default to True
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

            # Group-weighted, per-group PCA
            tprint("🔧 Using group weighting and per-group PCA for dimensionality reduction", "INFO")
            n_samples, n_features = features_scaled.shape
            n_components_total = min(20, n_features - 1)
            tprint(f"  📊 Input data: {n_samples} samples × {n_features} features", "DEBUG")
            tprint(f"  🎯 Total target components: {n_components_total}", "DEBUG")

            # Build group masks (returns/volatility/volume), leave others as passthrough block
            names = context.pre_pca_feature_names
            names_lower = [str(n).lower() for n in names]
            returns_mask = np.array([
                ('return' in name) or ('log_return' in name) or ('close_return' in name) for name in names_lower
            ], dtype=bool)
            volatility_mask = np.array([
                any(term in name for term in ['volatility', 'vol_', 'atr', 'std', 'boll', 'bb']) for name in names_lower
            ], dtype=bool)
            volume_mask = np.array([
                ('volume' in name) or ('vwap' in name) or ('obv' in name) or ('accumulation' in name) or ('distribution' in name) for name in names_lower
            ], dtype=bool)
            other_mask = ~(returns_mask | volatility_mask | volume_mask)

            # Weights per group (sqrt-applied so group variance ≈ weight)
            w_returns, w_vol, w_volume = 0.50, 0.30, 0.20
            features_w = features_scaled.copy()
            if np.any(returns_mask):
                features_w[:, returns_mask] *= np.sqrt(w_returns)
            if np.any(volatility_mask):
                features_w[:, volatility_mask] *= np.sqrt(w_vol)
            if np.any(volume_mask):
                features_w[:, volume_mask] *= np.sqrt(w_volume)

            # Allocate components per group proportional to weights and availability
            present = np.array([
                max(1, int(np.sum(returns_mask))),
                max(1, int(np.sum(volatility_mask))),
                max(1, int(np.sum(volume_mask)))
            ], dtype=int)
            weights = np.array([w_returns, w_vol, w_volume], dtype=float)
            weights = weights * (present > 0)
            if np.sum(weights) == 0:
                weights = np.array([1.0, 1.0, 1.0], dtype=float)
            weights /= np.sum(weights)
            allocated = np.maximum(1, np.round(weights * n_components_total).astype(int))
            # Clip to available dims
            allocated[0] = min(allocated[0], int(np.sum(returns_mask)))
            allocated[1] = min(allocated[1], int(np.sum(volatility_mask)))
            allocated[2] = min(allocated[2], int(np.sum(volume_mask)))
            # Ensure sum <= n_components_total
            while np.sum(allocated) > n_components_total:
                idx = np.argmax(allocated)
                if allocated[idx] > 1:
                    allocated[idx] -= 1
                else:
                    break

            def fit_group_pca(block: np.ndarray, ncomp: int):
                if block.shape[1] == 0:
                    return None, [], []
                if block.shape[1] == 1 or ncomp <= 1:
                    # passthrough or single-component PCA
                    comp = block if ncomp <= 0 else block[:, :1]
                    names_blk = ["PC1"] if comp.shape[1] == 1 else []
                    vars_blk = [1.0] if comp.shape[1] == 1 else []
                    return comp, names_blk, vars_blk
                ncomp = min(ncomp, block.shape[1])
                p = PCA(n_components=ncomp)
                comp = p.fit_transform(block)
                return comp, [f"PC{i+1}_var{p.explained_variance_ratio_[i]:.3f}" for i in range(ncomp)], list(p.explained_variance_ratio_)

            comps = []
            comp_names = []
            comp_scores = []
            # Returns
            if np.any(returns_mask):
                comp, names_blk, vars_blk = fit_group_pca(features_w[:, returns_mask], int(allocated[0]))
                if comp is not None and comp != []:
                    comps.append(comp)
                    comp_names += [f"RET_{n}" for n in names_blk] if names_blk else ["RET_PC1"]
                    comp_scores += (vars_blk if vars_blk else [1.0])
            # Volatility
            if np.any(volatility_mask):
                comp, names_blk, vars_blk = fit_group_pca(features_w[:, volatility_mask], int(allocated[1]))
                if comp is not None and comp != []:
                    comps.append(comp)
                    comp_names += [f"VOL_{n}" for n in names_blk] if names_blk else ["VOL_PC1"]
                    comp_scores += (vars_blk if vars_blk else [1.0])
            # Volume
            if np.any(volume_mask):
                comp, names_blk, vars_blk = fit_group_pca(features_w[:, volume_mask], int(allocated[2]))
                if comp is not None and comp != []:
                    comps.append(comp)
                    comp_names += [f"VLM_{n}" for n in names_blk] if names_blk else ["VLM_PC1"]
                    comp_scores += (vars_blk if vars_blk else [1.0])
            # Others: optional passthrough (no PCA) – cap to keep dimensionality reasonable
            if np.any(other_mask):
                # Keep at most 2 top-variance columns as-is
                other_block = features_w[:, other_mask]
                var_rank = np.argsort(-np.var(other_block, axis=0))
                keep = min(2, other_block.shape[1])
                passthrough = other_block[:, var_rank[:keep]] if keep > 0 else None
                if passthrough is not None and keep > 0:
                    comps.append(passthrough)
                    comp_names += [f"OTH_{i+1}" for i in range(keep)]
                    comp_scores += [1.0] * keep

            if len(comps) == 0:
                features_final = self._validate_feature_quality_minimal(features_scaled, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = list(feature_names)
                context.pca_loading_scores = {feature_names[i]: 1.0 for i in range(len(feature_names))}
            else:
                features_pca_group = np.concatenate(comps, axis=1)
                features_final = self._validate_feature_quality_minimal(features_pca_group, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = comp_names
                context.pca_loading_scores = {comp_names[i]: float(comp_scores[i]) for i in range(len(comp_names))}
                tprint(f"📈 Group PCA: RET={int(np.sum(returns_mask))}->{allocated[0]}, VOL={int(np.sum(volatility_mask))}->{allocated[1]}, VLM={int(np.sum(volume_mask))}->{allocated[2]}", "INFO")
                tprint(f"  📊 Final features: {features_final.shape[1]}", "INFO")

            if context.feature_scores:
                context.feature_scores = {n: float(context.pca_loading_scores.get(n, 1.0)) for n in context.optimized_feature_names}

            self._safe_memory_cleanup([features_scaled])

        except Exception as e:
            tprint(f"Feature optimization failed: {e}", "ERROR")
            raise ValueError(f"Feature optimization failed: {e}")

        return context

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

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature by its name to identify type (volatility, momentum, trend, etc.)."""
        feature_name_lower = feature_name.lower()

        # Volatility indicators
        if any(term in feature_name_lower for term in ['vol', 'volatility', 'atr', 'std', 'dev', 'range', 'bb', 'bollinger']):
            return "VOLATILITY"

        # Momentum indicators
        elif any(term in feature_name_lower for term in ['rsi', 'momentum', 'roc', 'rate_of_change', 'stoch', 'stochastic', 'williams', 'cci']):
            return "MOMENTUM"

        # Trend indicators
        elif any(term in feature_name_lower for term in ['ma', 'moving_average', 'ema', 'sma', 'trend', 'macd', 'adx', 'dmi', 'aroon']):
            return "TREND"

        # Volume indicators
        elif any(term in feature_name_lower for term in ['volume', 'vol', 'obv', 'ad', 'accumulation', 'distribution', 'mfi', 'money_flow']):
            return "VOLUME"

        # Price-based features
        elif any(term in feature_name_lower for term in ['price', 'close', 'open', 'high', 'low', 'return', 'change', 'pct']):
            return "PRICE"

        # Statistical features
        elif any(term in feature_name_lower for term in ['skew', 'kurt', 'stat', 'corr', 'correlation', 'beta', 'alpha']):
            return "STATISTICAL"

        # Regime features
        elif any(term in feature_name_lower for term in ['regime', 'state', 'phase', 'cycle']):
            return "REGIME"

        # Technical patterns
        elif any(term in feature_name_lower for term in ['pattern', 'signal', 'crossover', 'breakout', 'support', 'resistance']):
            return "PATTERN"

        # Default category
        else:
            return "OTHER"

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
