"""
Feature Selection Module for Market Analysis Clustering.

This module provides feature selection capabilities including:
- Near-zero variance feature elimination
- PCA loading score-based pruning
- Whitelist/blacklist feature filtering
- Financial feature domain knowledge integration

Rewritten from nas_tas_clustering tools without imports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, field
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import utility modules
from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.common_operations import get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
from ..shared import HardwareInitializer
from src.utils.common_utilities import safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from ..shared import safe_divide, ClusteringValidationUtils
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
# Lazy imports to avoid circular imports
def get_safe_correlation_matrix():
    """Lazy import of safe_correlation_matrix to avoid circular imports."""
    try:
        from src.utils.matrix_operations import safe_correlation_matrix
        return safe_correlation_matrix
    except ImportError:
        return None

def get_safe_matrix_multiply():
    """Lazy import of safe_matrix_multiply to avoid circular imports."""
    try:
        from src.utils.matrix_operations import safe_matrix_multiply
        return safe_matrix_multiply
    except ImportError:
        return None

def get_unified_matrix_operations():
    """Lazy import of get_unified_matrix_operations to avoid circular imports."""
    try:
        from src.utils.matrix_operations import get_unified_matrix_operations
        return get_unified_matrix_operations
    except ImportError:
        return None

# Hardware optimization imports
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_optimizer
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer as get_hw_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as get_hw_cpu_optimizer
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

@dataclass
class FeatureSelectorConfig:
    """Configuration for feature selection."""

    # Variance-based filtering
    variance_threshold: float = 0.01
    min_features: int = 10

    # PCA-based pruning
    pca_components: Optional[int] = None
    loading_threshold: float = 0.1
    max_features: int = 200

    # Domain-specific filtering
    financial_whitelist: List[str] = field(default_factory=list)
    financial_blacklist: List[str] = field(default_factory=list)

    # Selection strategy
    selection_method: str = "combined"  # "variance", "pca", "combined"
    target_features: int = 100

class FeatureSelector:
    """
    Feature selector for market analysis clustering.

    Responsibilities:
    - Drop near-zero-variance features
    - Prune by PCA loading scores
    - Allow whitelist/blacklist of financial features
    - Return final feature set
    """

    def __init__(self, config: FeatureSelectorConfig = None) -> None:
        """Initialize the FeatureSelector."""
        tprint("🚀 Initializing FeatureSelector", "INFO")
        self.config = config or FeatureSelectorConfig()
        self.selection_metadata: Dict[str, Any] = {}
        tprint_debug(f"Selector config: {self.config.__dict__}")

        # Initialize hardware optimization components
        tprint("⚡ Initializing hardware optimization components", "INFO")
        self._initialize_hardware_optimization()

        # Initialize matrix operations for efficient computations
        tprint("🔢 Initializing matrix operations", "INFO")
        self.matrix_ops = get_unified_matrix_operations(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True
        )
        tprint_debug("Matrix operations initialized with GPU and parallel processing")

        # Initialize HPO for parameter optimization
        tprint("🎯 Initializing hyperparameter optimization", "INFO")
        self.hpo_optimizer = HyperparameterOptimization()
        tprint_debug("HyperparameterOptimization initialized")

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components using shared utilities."""
        hardware_components = HardwareInitializer.initialize_hardware_components(
            "feature_selection", verbose=True
        )
        
        self.gpu_manager = hardware_components.get('gpu_manager')
        self.memory_optimizer = hardware_components.get('memory_manager')
        self.cpu_optimizer = hardware_components.get('cpu_optimizer')

    def select_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target_features: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Select features using the configured strategy.

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            target_features: Target number of features (overrides config)

        Returns:
            Tuple of (selected_features, selected_feature_names, metadata)
        """
        if target_features is not None:
            self.config.target_features = target_features

        tprint(f"🔍 FEATURE SELECTION: Starting with {features.shape[1]} features, target: {self.config.target_features}", color="cyan", bold=True)

        try:
            # Step 1: Apply domain-specific filtering (whitelist/blacklist)
            features, feature_names = self._apply_domain_filtering(features, feature_names)

            # Step 2: Drop near-zero variance features
            features, feature_names = self._drop_near_zero_variance(features, feature_names)

            # Step 3: Apply PCA-based pruning if using PCA or combined method
            if self.config.selection_method in ["pca", "combined"]:
                features, feature_names = self._apply_pca_pruning(features, feature_names)

            # Step 4: Final variance-based selection if needed
            if self.config.selection_method in ["variance", "combined"]:
                features, feature_names = self._apply_final_variance_selection(features, feature_names)

            # Store metadata
            self.selection_metadata = {
                'original_features': features.shape[1],
                'selected_features': len(feature_names),
                'selection_method': self.config.selection_method,
                'target_features': self.config.target_features,
                'remaining_features': feature_names
            }

            tprint(f"✅ Feature selection completed: {features.shape[1]} features selected", "SUCCESS")

            return features, feature_names, self.selection_metadata

        except Exception as e:
            tprint(f"❌ Feature selection failed: {e}", "ERROR")
            return features, feature_names, {'error': str(e)}

    def _apply_domain_filtering(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply domain-specific whitelist/blacklist filtering."""
        if not self.config.financial_whitelist and not self.config.financial_blacklist:
            return features, feature_names

        tprint("🔍 Applying domain-specific feature filtering...", "INFO")

        # Create filtering mask
        mask = np.ones(len(feature_names), dtype=bool)

        # Apply whitelist (if specified)
        if self.config.financial_whitelist:
            whitelist_mask = np.isin(feature_names, self.config.financial_whitelist)
            mask = mask & whitelist_mask
            tprint(f"   • Whitelist: {whitelist_mask.sum()}/{len(feature_names)} features retained", "INFO")

        # Apply blacklist (if specified)
        if self.config.financial_blacklist:
            blacklist_mask = ~np.isin(feature_names, self.config.financial_blacklist)
            mask = mask & blacklist_mask
            tprint(f"   • Blacklist: {len(feature_names) - blacklist_mask.sum()}/{len(feature_names)} features removed", "INFO")

        # Filter features
        filtered_features = features[:, mask]
        filtered_names = [name for name, keep in zip(feature_names, mask) if keep]

        tprint(f"   • Domain filtering: {filtered_features.shape[1]}/{features.shape[1]} features retained", "INFO")

        return filtered_features, filtered_names

    def _drop_near_zero_variance(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Drop features with near-zero variance."""
        tprint("🔍 Dropping near-zero variance features...", "INFO")

        # Validate input features
        features = validate_finite(features, "features")

        # Calculate variance for each feature using matrix operations
        try:
            variances = self.matrix_ops.batch_feature_transformation(
                features,
                lambda x: np.var(x, axis=0),
                operation_name="variance_calculation"
            )
        except Exception:
            # Fallback to standard numpy calculation
            variances = np.var(features, axis=0)

        # Validate variance threshold
        variance_threshold = validate_positive(self.config.variance_threshold, "variance_threshold")

        # Identify features to keep (variance > threshold)
        keep_mask = variances > variance_threshold

        # Ensure we don't drop too many features
        if keep_mask.sum() < self.config.min_features:
            # Keep top features by variance using safe sorting
            try:
                # Use matrix operations for sorting if available
                sorted_indices = self.matrix_ops.get_batch_matrix_processor().argsort_batch(
                    variances.reshape(1, -1), descending=True
                )[0]
            except Exception:
                # Fallback to numpy
                sorted_indices = np.argsort(variances)[::-1]

            top_indices = sorted_indices[:self.config.min_features]
            keep_mask = np.zeros_like(keep_mask, dtype=bool)
            keep_mask[top_indices] = True
            tprint(f"   • Warning: Only {keep_mask.sum()} features above threshold, keeping top {self.config.min_features}", "WARNING")

        # Filter features using matrix operations for efficiency
        try:
            filtered_features = self.matrix_ops.batch_feature_transformation(
                features,
                lambda x: x[:, keep_mask],
                operation_name="feature_filtering"
            )
        except Exception:
            filtered_features = features[:, keep_mask]

        filtered_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

        # Calculate variance statistics safely
        min_var = safe_divide(variances.min(), 1.0, 0.0)
        max_var = safe_divide(variances.max(), 1.0, 0.0)
        mean_var = safe_divide(variances.mean(), 1.0, 0.0)

        tprint(f"   • Variance filtering: {filtered_features.shape[1]}/{features.shape[1]} features retained", "INFO")
        tprint(f"   • Variance stats - Min: {min_var:.6f}, Max: {max_var:.6f}, Mean: {mean_var:.6f}", "INFO")

        return filtered_features, filtered_names

    def _apply_pca_pruning(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply PCA-based pruning using loading scores."""
        tprint("🔍 Applying PCA-based feature pruning...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")

        # Standardize features before PCA using matrix operations
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        except Exception as e:
            tprint(f"⚠️ StandardScaler failed, using matrix operations: {e}", "WARNING")
            # Use matrix operations for scaling
            mean_vals = np.mean(features, axis=0)
            std_vals = np.std(features, axis=0)
            # Avoid division by zero
            std_vals = np.where(std_vals == 0, 1e-8, std_vals)
            features_scaled = (features - mean_vals) / std_vals

        # Determine number of PCA components using validation
        n_samples, n_features = features_scaled.shape

        if self.config.pca_components is None:
            # Use min of n_samples/10, n_features/2, or max_features
            max_components = min(
                validate_positive(n_samples // 10, "max_components_samples"),
                validate_positive(n_features // 2, "max_components_features"),
                validate_positive(self.config.max_features, "max_features")
            )
            n_components = min(max_components, validate_positive(self.config.target_features, "target_features"))
        else:
            pca_components = validate_positive(self.config.pca_components, "pca_components")
            n_components = min(pca_components, n_features)

        # Apply PCA using optimized matrix operations
        try:
            # Use matrix operations for PCA computation if available
            pca_result = self.matrix_ops.matrix_correlation_analysis(
                features_scaled,
                n_components=n_components,
                method='pca'
            )
            features_pca = pca_result['transformed_data']
            pca = pca_result.get('pca_model')
        except Exception:
            # Fallback to standard PCA
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)

        # Compute loading scores using safe operations
        loading_scores = self._compute_loading_scores(pca, features_scaled.shape[1])

        # Validate loading threshold
        loading_threshold = validate_positive(self.config.loading_threshold, "loading_threshold")

        # Prune features based on loading scores
        if loading_scores.size > self.config.target_features:
            tprint(f"   • Pruning {loading_scores.size} features to {self.config.target_features} using loading scores", "INFO")

            # Sort by loading scores and keep top features using safe operations
            try:
                sorted_indices = np.argsort(loading_scores)[::-1]
            except Exception:
                # Fallback if sorting fails
                sorted_indices = np.arange(len(loading_scores))

            retained_mask = np.zeros_like(loading_scores, dtype=bool)

            # Ensure we keep at least 2 features
            target_features = validate_positive(self.config.target_features, "target_features")
            n_to_keep = min(target_features, len(sorted_indices))
            retained_mask[sorted_indices[:n_to_keep]] = True

            # If no features retained (shouldn't happen), keep top 2
            if retained_mask.sum() < 2 and loading_scores.size >= 2:
                retained_mask = np.zeros_like(loading_scores, dtype=bool)
                retained_mask[sorted_indices[:2]] = True

            # If still no features retained, keep all features above threshold
            if retained_mask.sum() == 0:
                retained_mask = loading_scores > loading_threshold

            # If still no features, keep the highest scoring one
            if retained_mask.sum() == 0 and loading_scores.size > 0:
                retained_mask[np.argmax(loading_scores)] = True

        else:
            retained_mask = loading_scores > loading_threshold

        # Apply filtering using matrix operations
        try:
            filtered_features = features[:, retained_mask]
        except Exception:
            filtered_features = features[:, retained_mask]

        filtered_names = [name for name, keep in zip(feature_names, retained_mask) if keep]

        # Calculate statistics safely
        max_loading = safe_divide(loading_scores.max(), 1.0, 0.0)

        tprint(f"   • PCA pruning: {filtered_features.shape[1]}/{features.shape[1]} features retained", "INFO")
        tprint(f"   • Loading threshold: {loading_threshold}, max loading: {max_loading:.6f}", "INFO")

        return filtered_features, filtered_names

    def _apply_final_variance_selection(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply final variance-based selection if needed."""
        if features.shape[1] <= self.config.target_features:
            return features, feature_names

        tprint("🔍 Applying final variance-based selection...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")
        target_features = validate_positive(self.config.target_features, "target_features")

        # Calculate variance for each feature using matrix operations
        try:
            variances = self.matrix_ops.batch_feature_transformation(
                features,
                lambda x: np.var(x, axis=0),
                operation_name="final_variance_calculation"
            )
        except Exception:
            variances = np.var(features, axis=0)

        # Select top features by variance using safe operations
        try:
            # Use matrix operations for efficient sorting
            sorted_indices = self.matrix_ops.get_batch_matrix_processor().argsort_batch(
                variances.reshape(1, -1), descending=True
            )[0]
        except Exception:
            sorted_indices = np.argsort(variances)[::-1]

        top_indices = sorted_indices[:target_features]

        # Apply selection using matrix operations
        try:
            selected_features = features[:, top_indices]
        except Exception:
            selected_features = features[:, top_indices]

        selected_names = [feature_names[i] for i in top_indices]

        tprint(f"   • Variance selection: {selected_features.shape[1]}/{features.shape[1]} features retained", "INFO")

        return selected_features, selected_names

    def _compute_loading_scores(self, pca_model: PCA, n_features: int) -> np.ndarray:
        """Compute loading scores for PCA components."""
        try:
            # Get the components (loadings) using safe matrix operations
            components = pca_model.components_

            # Validate components
            components = validate_finite(components, "pca_components")

            # Compute loading scores as the sum of absolute values across all components
            # This gives us an overall importance score for each feature
            try:
                # Use matrix operations for efficient computation
                abs_components = self.matrix_ops.batch_feature_transformation(
                    components,
                    lambda x: np.abs(x),
                    operation_name="absolute_components"
                )
                loading_scores = np.sum(abs_components, axis=0)
            except Exception:
                # Fallback to standard computation
                loading_scores = np.sum(np.abs(components), axis=0)

            return loading_scores

        except Exception as e:
            tprint(f"⚠️ Loading score computation failed: {e}", "WARNING")
            # Return uniform scores as fallback
            return np.ones(n_features)

    def get_selection_metadata(self) -> Dict[str, Any]:
        """Get the selection metadata."""
        return self.selection_metadata.copy()

    def add_financial_features_to_whitelist(self, features: List[str]) -> None:
        """Add features to the financial whitelist."""
        self.config.financial_whitelist.extend(features)

    def add_financial_features_to_blacklist(self, features: List[str]) -> None:
        """Add features to the financial blacklist."""
        self.config.financial_blacklist.extend(features)
