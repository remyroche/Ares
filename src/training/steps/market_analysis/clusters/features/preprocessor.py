"""
Feature Preprocessing Module for Market Analysis Clustering.

This module provides feature preprocessing capabilities including:
- Scaling (RobustScaler, MinMax)
- NaN and outlier handling
- Standardization
- PCA/UMAP dimensionality reduction
- Integration with src/utils/data/ tools

Uses tools from src/utils/data/ for preprocessing operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Import utilities from src/utils/data/
try:
    from src.utils.data.processing.data_processing import DataProcessor
    from src.utils.data.processing.transformers import DataTransformer
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

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
from ..shared import ClusteringValidationUtils, ClusteringCommonUtils, safe_divide
from src.utils.matrix_operations import get_unified_matrix_operations, safe_matrix_multiply, safe_correlation_matrix

# Hardware optimization imports
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_optimizer
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer as get_hw_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as get_hw_cpu_optimizer
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

@dataclass
class FeaturePreprocessorConfig:
    """Configuration for feature preprocessing."""

    # Scaling options
    scaling_method: str = "robust"  # "robust", "minmax", "standard", "none"
    scaling_params: Dict[str, Any] = field(default_factory=dict)

    # NaN handling
    nan_strategy: str = "median"  # "mean", "median", "most_frequent", "constant"
    nan_fill_value: float = 0.0

    # Outlier handling
    outlier_method: str = "clip"  # "clip", "remove", "winsorize", "none"
    outlier_threshold: float = 3.0  # Standard deviations or IQR multiplier

    # Dimensionality reduction
    reduction_method: str = "pca"  # "pca", "umap", "none"
    n_components: Optional[int] = None
    reduction_params: Dict[str, Any] = field(default_factory=dict)

    # Data quality
    remove_duplicates: bool = True
    handle_infinity: bool = True
    validate_features: bool = True

class FeaturePreprocessor:
    """
    Feature preprocessor for market analysis clustering.

    Responsibilities:
    - Scale (RobustScaler, MinMax if needed)
    - Handle NaNs, outliers, standardization
    - Apply PCA/UMAP dimensionality reduction
    - Return reduced, clean matrix
    """

    def __init__(self, config: FeaturePreprocessorConfig = None):
        """Initialize the FeaturePreprocessor."""
        self.config = config or FeaturePreprocessorConfig()
        self.preprocessing_metadata = {}

        # Initialize hardware optimization components
        self._initialize_hardware_optimization()

        # Initialize matrix operations for efficient computations
        self.matrix_ops = get_unified_matrix_operations(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True
        )

        # Initialize data processing utilities if available
        if DATA_PROCESSING_AVAILABLE:
            self.data_processor = DataProcessor()
            self.data_transformer = DataTransformer()
        else:
            self.data_processor = None
            self.data_transformer = None
            tprint("⚠️ Data processing utilities not available, using sklearn fallbacks", "WARNING")

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components using shared utilities."""
        hardware_components = HardwareInitializer.initialize_hardware_components(
            "feature_preprocessing", verbose=True
        )
        
        self.gpu_manager = hardware_components.get('gpu_manager')
        self.memory_optimizer = hardware_components.get('memory_manager')
        self.cpu_optimizer = hardware_components.get('cpu_optimizer')

    def preprocess_features(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Preprocess features using the configured pipeline.

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: List of feature names

        Returns:
            Tuple of (processed_features, processed_feature_names, metadata)
        """
        tprint(f"🔧 FEATURE PREPROCESSING: Starting with {features.shape[1]} features, {features.shape[0]} samples", color="cyan", bold=True)

        try:
            # Step 1: Data quality checks and cleaning
            features, feature_names = self._clean_data(features, feature_names)

            # Step 2: Handle NaN values
            features = self._handle_nans(features)

            # Step 3: Handle outliers
            features = self._handle_outliers(features)

            # Step 4: Apply scaling
            features, scaler = self._apply_scaling(features)

            # Step 5: Apply dimensionality reduction
            features, feature_names, reducer = self._apply_dimensionality_reduction(features, feature_names)

            # Store metadata
            self.preprocessing_metadata = {
                'original_shape': features.shape,
                'processed_shape': features.shape,
                'scaling_method': self.config.scaling_method,
                'nan_strategy': self.config.nan_strategy,
                'outlier_method': self.config.outlier_method,
                'reduction_method': self.config.reduction_method,
                'n_components': features.shape[1] if features.ndim > 1 else 1,
                'remaining_features': feature_names
            }

            tprint(f"✅ Feature preprocessing completed: {features.shape[1]} features processed", "SUCCESS")

            return features, feature_names, self.preprocessing_metadata

        except Exception as e:
            tprint(f"❌ Feature preprocessing failed: {e}", "ERROR")
            return features, feature_names, {'error': str(e)}

    def _clean_data(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Clean data by removing duplicates and handling infinity values."""
        tprint("🔍 Cleaning data...", "INFO")

        # Remove duplicates if requested
        if self.config.remove_duplicates:
            # Check for duplicate rows
            unique_indices = ~pd.DataFrame(features).duplicated().values
            if unique_indices.sum() < features.shape[0]:
                features = features[unique_indices]
                tprint(f"   • Removed {features.shape[0] - unique_indices.sum()} duplicate rows", "INFO")

        # Handle infinity values if requested
        if self.config.handle_infinity:
            inf_mask = np.isinf(features)
            if inf_mask.any():
                tprint(f"   • Found {inf_mask.sum()} infinite values, replacing with NaN", "WARNING")
                features = np.where(inf_mask, np.nan, features)

        # Validate features if requested
        if self.config.validate_features:
            self._validate_features(features, feature_names)

        return features, feature_names

    def _handle_nans(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values in the feature matrix."""
        tprint("🔍 Handling NaN values...", "INFO")

        # Validate input features using shared utilities
        validation_result = ClusteringValidationUtils.validate_features(features)
        if not validation_result.is_valid:
            raise ValueError(f"Feature validation failed: {validation_result.errors}")
        features = validate_finite(features, "features")

        nan_mask = np.isnan(features)
        total_nans = nan_mask.sum()

        if total_nans == 0:
            tprint("   • No NaN values found", "INFO")
            return features

        # Calculate percentage safely
        nan_percentage = safe_divide(total_nans * 100, features.size, 0.0)

        tprint(f"   • Found {total_nans} NaN values ({nan_percentage:.2f}%)", "INFO")

        # Use sklearn imputer for NaN handling with validation
        try:
            imputer = SimpleImputer(
                strategy=self.config.nan_strategy,
                fill_value=self.config.nan_fill_value
            )

            # Reshape for imputer (it expects 2D)
            original_shape = features.shape
            if features.ndim == 1:
                features = features.reshape(-1, 1)

            features = imputer.fit_transform(features)

            # Restore original shape if needed
            if len(original_shape) == 1 and original_shape[0] == features.shape[0]:
                features = features.ravel()

        except Exception as e:
            tprint(f"⚠️ sklearn imputer failed, using fallback: {e}", "WARNING")
            # Fallback to simple imputation
            if self.config.nan_strategy == "median":
                fill_values = np.nanmedian(features, axis=0)
            elif self.config.nan_strategy == "mean":
                fill_values = np.nanmean(features, axis=0)
            else:
                fill_values = self.config.nan_fill_value

            # Fill NaN values
            for i in range(features.shape[1]):
                nan_indices = np.isnan(features[:, i])
                if nan_indices.any():
                    features[nan_indices, i] = fill_values[i] if hasattr(fill_values, '__getitem__') else fill_values

        tprint(f"   • NaN handling complete using {self.config.nan_strategy} strategy", "SUCCESS")

        return features

    def _handle_outliers(self, features: np.ndarray) -> np.ndarray:
        """Handle outliers in the feature matrix."""
        if self.config.outlier_method == "none":
            return features

        tprint(f"🔍 Handling outliers using {self.config.outlier_method} method...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")
        outlier_threshold = validate_positive(self.config.outlier_threshold, "outlier_threshold")

        if self.config.outlier_method == "clip":
            # Clip outliers based on standard deviation using matrix operations
            try:
                mean_vals = self.matrix_ops.batch_feature_transformation(
                    features,
                    lambda x: np.mean(x, axis=0),
                    operation_name="mean_calculation"
                )
                std_vals = self.matrix_ops.batch_feature_transformation(
                    features,
                    lambda x: np.std(x, axis=0),
                    operation_name="std_calculation"
                )

                # Avoid division by zero using safe operations
                std_vals = np.where(std_vals == 0, 1e-8, std_vals)

                lower_bound = mean_vals - outlier_threshold * std_vals
                upper_bound = mean_vals + outlier_threshold * std_vals

                # Use matrix operations for clipping
                features = self.matrix_ops.batch_feature_transformation(
                    features,
                    lambda x: np.clip(x, lower_bound, upper_bound),
                    operation_name="clipping"
                )
            except Exception as e:
                tprint(f"⚠️ Matrix operations clipping failed, using fallback: {e}", "WARNING")
                # Fallback to standard numpy operations
                mean_vals = np.mean(features, axis=0)
                std_vals = np.std(features, axis=0)
                std_vals = np.where(std_vals == 0, 1e-8, std_vals)
                lower_bound = mean_vals - outlier_threshold * std_vals
                upper_bound = mean_vals + outlier_threshold * std_vals
                features = np.clip(features, lower_bound, upper_bound)

        elif self.config.outlier_method == "remove":
            # Remove outlier rows (this is aggressive and might remove too much data)
            tprint("   • Warning: Removing outlier rows may significantly reduce dataset size", "WARNING")

            # Calculate z-scores using matrix operations
            try:
                mean_vals = self.matrix_ops.batch_feature_transformation(
                    features,
                    lambda x: np.mean(x, axis=0),
                    operation_name="mean_calculation"
                )
                std_vals = self.matrix_ops.batch_feature_transformation(
                    features,
                    lambda x: np.std(x, axis=0),
                    operation_name="std_calculation"
                )
                std_vals = np.where(std_vals == 0, 1e-8, std_vals)

                z_scores = self.matrix_ops.batch_feature_transformation(
                    features,
                    lambda x: np.abs((x - mean_vals) / std_vals),
                    operation_name="z_score_calculation"
                )
                outlier_mask = np.any(z_scores > outlier_threshold, axis=1)

                original_rows = features.shape[0]
                features = features[~outlier_mask]

                removed_count = original_rows - features.shape[0]
                tprint(f"   • Removed {removed_count}/{original_rows} outlier rows", "INFO")

            except Exception as e:
                tprint(f"⚠️ Matrix operations outlier removal failed, using fallback: {e}", "WARNING")
                # Fallback to standard operations
                mean_vals = np.mean(features, axis=0)
                std_vals = np.std(features, axis=0)
                std_vals = np.where(std_vals == 0, 1e-8, std_vals)
                z_scores = np.abs((features - mean_vals) / std_vals)
                outlier_mask = np.any(z_scores > outlier_threshold, axis=1)
                features = features[~outlier_mask]

        elif self.config.outlier_method == "winsorize":
            # Winsorize outliers (clip to percentiles) using matrix operations
            try:
                for i in range(features.shape[1]):
                    col = features[:, i]
                    if np.std(col) > 0:  # Avoid division by zero for constant columns
                        lower_percentile = np.percentile(col, 5)
                        upper_percentile = np.percentile(col, 95)

                        # Use safe clipping
                        features[:, i] = np.clip(col, lower_percentile, upper_percentile)
            except Exception as e:
                tprint(f"⚠️ Winsorization failed, using fallback: {e}", "WARNING")
                # Fallback for each column
                for i in range(features.shape[1]):
                    col = features[:, i]
                    if np.std(col) > 0:
                        lower_percentile = np.percentile(col, 5)
                        upper_percentile = np.percentile(col, 95)
                        features[:, i] = np.clip(col, lower_percentile, upper_percentile)

        tprint(f"   • Outlier handling complete using {self.config.outlier_method} method", "SUCCESS")

        return features

    def _apply_scaling(self, features: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Apply scaling to the feature matrix."""
        if self.config.scaling_method == "none":
            return features, None

        tprint(f"🔍 Applying {self.config.scaling_method} scaling...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")

        if self.config.scaling_method == "robust":
            scaler = RobustScaler(**self.config.scaling_params)
        elif self.config.scaling_method == "minmax":
            scaler = MinMaxScaler(**self.config.scaling_params)
        elif self.config.scaling_method == "standard":
            scaler = StandardScaler(**self.config.scaling_params)
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")

        # Reshape for scaler if needed
        original_shape = features.shape
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        try:
            features = scaler.fit_transform(features)
        except Exception as e:
            tprint(f"⚠️ sklearn scaler failed, using manual scaling: {e}", "WARNING")
            # Manual scaling as fallback
            if self.config.scaling_method == "robust":
                # Robust scaling: (x - median) / IQR
                median_vals = np.median(features, axis=0)
                q75, q25 = np.percentile(features, [75, 25], axis=0)
                iqr = q75 - q25
                iqr = np.where(iqr == 0, 1e-8, iqr)  # Avoid division by zero
                features = (features - median_vals) / iqr
            elif self.config.scaling_method == "minmax":
                # MinMax scaling: (x - min) / (max - min)
                min_vals = np.min(features, axis=0)
                max_vals = np.max(features, axis=0)
                range_vals = max_vals - min_vals
                range_vals = np.where(range_vals == 0, 1e-8, range_vals)  # Avoid division by zero
                features = (features - min_vals) / range_vals
            elif self.config.scaling_method == "standard":
                # Standard scaling: (x - mean) / std
                mean_vals = np.mean(features, axis=0)
                std_vals = np.std(features, axis=0)
                std_vals = np.where(std_vals == 0, 1e-8, std_vals)  # Avoid division by zero
                features = (features - mean_vals) / std_vals

        # Restore original shape if needed
        if len(original_shape) == 1 and original_shape[0] == features.shape[0]:
            features = features.ravel()

        tprint(f"   • Scaling complete using {self.config.scaling_method} method", "SUCCESS")

        return features, scaler

    def _apply_dimensionality_reduction(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Any]:
        """Apply dimensionality reduction."""
        if self.config.reduction_method == "none":
            return features, feature_names, None

        tprint(f"🔍 Applying {self.config.reduction_method} dimensionality reduction...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")

        if self.config.reduction_method == "pca":
            # Determine number of components using validation
            n_features = features.shape[1]
            if self.config.n_components is None:
                n_components = min(n_features, 50)  # Default to 50 or max available
            else:
                n_components = min(validate_positive(self.config.n_components, "n_components"), n_features)

            try:
                # Use matrix operations for PCA if available
                pca_result = self.matrix_ops.matrix_correlation_analysis(
                    features,
                    n_components=n_components,
                    method='pca',
                    **self.config.reduction_params
                )
                features_reduced = pca_result['transformed_data']
                reducer = pca_result.get('pca_model', PCA(n_components=n_components))
                explained_variance = pca_result.get('explained_variance_ratio', np.ones(n_components) / n_components)
            except Exception as e:
                tprint(f"⚠️ Matrix operations PCA failed, using sklearn fallback: {e}", "WARNING")
                # Fallback to standard PCA
                reducer = PCA(n_components=n_components, **self.config.reduction_params)
                features_reduced = reducer.fit_transform(features)
                explained_variance = reducer.explained_variance_ratio_

            # Create new feature names
            reduced_names = [f"pca_{i}" for i in range(features_reduced.shape[1])]

            # Calculate explained variance safely
            total_explained = safe_divide(np.sum(explained_variance), 1.0, 0.0)

            tprint(f"   • PCA reduction: {features.shape[1]} → {features_reduced.shape[1]} components", "INFO")
            tprint(f"   • Explained variance: {total_explained:.4f}", "INFO")

        elif self.config.reduction_method == "umap":
            if not UMAP_AVAILABLE:
                tprint("   • UMAP not available, falling back to PCA", "WARNING")
                return self._apply_dimensionality_reduction(features, feature_names)

            # Determine number of components using validation
            n_features = features.shape[1]
            if self.config.n_components is None:
                n_components = min(n_features, 20)  # Default to 20 for UMAP
            else:
                n_components = min(validate_positive(self.config.n_components, "n_components"), n_features)

            try:
                reducer = umap.UMAP(
                    n_components=n_components,
                    **self.config.reduction_params
                )
                features_reduced = reducer.fit_transform(features)
            except Exception as e:
                tprint(f"⚠️ UMAP failed, falling back to PCA: {e}", "WARNING")
                # Fallback to PCA
                return self._apply_dimensionality_reduction(features, feature_names)

            # Create new feature names
            reduced_names = [f"umap_{i}" for i in range(features_reduced.shape[1])]

            tprint(f"   • UMAP reduction: {features.shape[1]} → {features_reduced.shape[1]} components", "INFO")

        else:
            raise ValueError(f"Unknown reduction method: {self.config.reduction_method}")

        return features_reduced, reduced_names, reducer

    def _validate_features(self, features: np.ndarray, feature_names: List[str]) -> None:
        """Validate feature matrix for common issues."""
        issues = []

        # Check for all-NaN features
        nan_features = np.isnan(features).all(axis=0)
        if nan_features.any():
            issues.append(f"All-NaN features: {np.where(nan_features)[0]}")

        # Check for constant features (zero variance)
        constant_features = np.var(features, axis=0) == 0
        if constant_features.any():
            issues.append(f"Constant features: {np.where(constant_features)[0]}")

        # Check for infinite values
        inf_features = np.isinf(features)
        if inf_features.any():
            issues.append(f"Features with infinite values: {np.where(inf_features.any(axis=0))[0]}")

        if issues:
            tprint(f"   • Validation issues found: {'; '.join(issues)}", "WARNING")
        else:
            tprint("   • Feature validation passed", "SUCCESS")

    def get_preprocessing_metadata(self) -> Dict[str, Any]:
        """Get the preprocessing metadata."""
        return self.preprocessing_metadata.copy()

    def fit_transform_features(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Alias for preprocess_features for sklearn compatibility."""
        return self.preprocess_features(features, feature_names)
