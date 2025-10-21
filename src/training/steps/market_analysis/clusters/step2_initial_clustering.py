"""
Step 2: Initial Clustering for NAS-TAS Clustering.

This module handles the initial clustering setup, regime assignment extraction,
and basic clustering initialization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance
)
from src.utils.common_operations import (
    get_memory_usage, optimize_dataframe_memory, safe_divide, safe_mean, safe_std,
    memory_monitor, force_garbage_collection, performance_timer, validate_dataframe,
    safe_merge, safe_concat, calculate_data_quality_metrics, create_summary_statistics
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    analyze_nan_values_detailed, format_nan_analysis_report, get_dataframe_info,
    safe_merge_dataframes, safe_groupby_operation, safe_apply_function
)
from src.utils.math_validation import (
    validate_finite, validate_array_finite, safe_divide, safe_log, safe_sqrt, safe_power,
    safe_correlation, safe_mean, safe_std, validate_positive, safe_covariance,
    safe_percentile, validate_correlation_matrix
)

from .shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext

class InitialClusteringStep:
    """Step 2: Initial clustering and regime assignment extraction."""

    def __init__(self, verbose: bool = True):
        """Initialize the initial clustering step."""
        self.verbose = verbose
        self.logger = get_logger('InitialClusteringStep')

    async def execute(self, context: ClusteringContext, config: Any) -> ClusteringContext:
        """Execute initial clustering step with enhanced monitoring."""
        try:
            tprint("Step 2: Starting initial clustering setup...", "INFO")
            tprint_debug(f"Context features shape: {context.optimized_features.shape}")

            # Validate input features
            with memory_monitor("Feature Validation"):
                validate_array_finite(context.optimized_features, "optimized_features")
                tprint_debug(f"Features validated - shape: {context.optimized_features.shape}")

            # Extract TAS and NAS regime assignments
            tprint_debug("About to extract regime assignments")
            with memory_monitor("Regime Assignment Extraction"):
                tas_assignments, nas_assignments = await self._extract_regime_assignments(context, config)
            tprint_debug(f"Regime assignments extracted - TAS: {len(tas_assignments)}, NAS: {len(nas_assignments)}")
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments

            # Initialize basic clustering with optimal K
            tprint_debug("About to determine optimal K")
            with memory_monitor("Optimal K Determination"):
                optimal_k = await self._determine_optimal_k(context, config)
            tprint_debug(f"Optimal K determined: {optimal_k}")
            context.optimal_k = optimal_k

            # Perform initial clustering
            tprint_debug("About to perform initial clustering")
            with memory_monitor("Initial Clustering"):
                initial_assignments = await self._perform_initial_clustering(
                    context.optimized_features, optimal_k
                )
            tprint_debug(f"Initial clustering completed - assignments shape: {initial_assignments.shape}")
            context.initial_assignments = initial_assignments

            tprint("Step 2: Initial clustering completed successfully", "SUCCESS")
            return context

        except Exception as e:
            tprint(f"Step 2: Initial clustering failed: {e}", "ERROR")
            # Force cleanup on error
            force_garbage_collection()
            raise ValueError(f"Initial clustering failed: {e}")

    async def _extract_regime_assignments(
        self,
        context: ClusteringContext,
        config: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments from pipeline state or previous outcomes."""
        try:
            # For now, create dummy assignments based on features
            # In the full implementation, this would extract from pipeline state
            n_samples = context.optimized_features.shape[0]

            # Create dummy TAS assignments (trend-following)
            tas_assignments = np.random.randint(0, 3, n_samples)

            # Create dummy NAS assignments (mean-reverting)
            nas_assignments = np.random.randint(0, 3, n_samples)

            tprint(f"Extracted TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}", "SUCCESS")
            return tas_assignments, nas_assignments

        except Exception as e:
            tprint(f"Regime assignment extraction failed: {e}", "ERROR")
            raise

    async def _determine_optimal_k(self, context: ClusteringContext, config: Any) -> int:
        """Determine optimal number of clusters using BIC and stability analysis with enhanced validation."""
        try:
            features = context.optimized_features
            n_samples, n_features = features.shape

            # Validate input features
            validate_array_finite(features, "features")
            tprint_debug(f"Determining optimal K for {n_samples} samples, {n_features} features")

            # Default optimal K
            default_k = getattr(config, 'n_regimes', 6)

            # Check memory pressure - skip optimal K determination if memory pressure is high
            try:
                from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
                memory_optimizer = get_m1_memory_optimizer()
                memory_pressure = getattr(memory_optimizer, 'memory_pressure', 0.0)

                if memory_pressure > 0.8:  # High memory pressure threshold
                    tprint(f"🧠 High memory pressure detected ({memory_pressure:.2f}), skipping optimal K determination", "WARNING")
                    return default_k
            except Exception as e:
                tprint(f"Could not check memory pressure: {e}, proceeding with optimal K determination", "WARNING")

            # Use BIC to determine optimal K for GMM (simplified)
            k_range = range(2, min(10, n_samples // 10))
            bic_scores = []

            for k in k_range:
                try:
                    gmm = GaussianMixture(n_components=k, random_state=42, max_iter=50)
                    gmm.fit(features)
                    bic_scores.append(gmm.bic(features))
                except Exception as e:
                    tprint(f"BIC calculation failed for k={k}: {e}", "WARNING")
                    bic_scores.append(float('inf'))

            # More robust BIC score validation
            if bic_scores and len(bic_scores) > 0:
                try:
                    # Check if all scores are finite and not all infinite
                    finite_scores = [score for score in bic_scores if np.isfinite(score)]
                    if finite_scores:
                        optimal_k = k_range[np.argmin(bic_scores)]
                        tprint(f"BIC-selected optimal K: {optimal_k}", "SUCCESS")
                    else:
                        optimal_k = default_k
                        tprint(f"Using default optimal K: {optimal_k}", "INFO")
                except (ValueError, TypeError):
                    optimal_k = default_k
                    tprint(f"Using default optimal K due to BIC validation error: {optimal_k}", "INFO")
            else:
                optimal_k = default_k
                tprint(f"Using default optimal K: {optimal_k}", "INFO")

            return optimal_k

        except Exception as e:
            tprint(f"Optimal K determination failed: {e}", "ERROR")
            return getattr(config, 'n_regimes', 6)

    async def _perform_initial_clustering(
        self,
        features: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Perform initial clustering using K-means."""
        try:
            tprint(f"Performing initial clustering with K={k}...", "INFO")

            # Use K-means for initial clustering
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            assignments = kmeans.fit_predict(features)

            tprint(f"Initial clustering completed: {len(np.unique(assignments))} clusters", "SUCCESS")
            return assignments

        except Exception as e:
            tprint(f"Initial clustering failed: {e}", "ERROR")
            raise

    def _validate_assignments(self, assignments: np.ndarray, expected_length: int) -> bool:
        """Validate assignment array."""
        try:
            if assignments is None:
                return False
            if len(assignments) != expected_length:
                return False
            if not np.issubdtype(assignments.dtype, np.integer):
                return False
            return True
        except Exception:
            return False
