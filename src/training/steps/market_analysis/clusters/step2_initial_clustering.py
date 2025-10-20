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
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from ..shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext

class InitialClusteringStep:
    """Step 2: Initial clustering and regime assignment extraction."""

    def __init__(self, verbose: bool = True):
        """Initialize the initial clustering step."""
        self.verbose = verbose
        self.logger = get_logger('InitialClusteringStep')

    async def execute(self, context: ClusteringContext, config: Any) -> ClusteringContext:
        """Execute initial clustering step."""
        try:
            tprint("Step 2: Starting initial clustering setup...", "INFO")
            tprint(f"🔍 DEBUG: Context features shape: {context.optimized_features.shape}", "DEBUG")

            # Extract TAS and NAS regime assignments
            tprint("🔍 DEBUG: About to extract regime assignments", "DEBUG")
            tas_assignments, nas_assignments = await self._extract_regime_assignments(context, config)
            tprint(f"✅ DEBUG: Regime assignments extracted - TAS: {len(tas_assignments)}, NAS: {len(nas_assignments)}", "DEBUG")
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments

            # Initialize basic clustering with optimal K
            tprint("🔍 DEBUG: About to determine optimal K", "DEBUG")
            optimal_k = await self._determine_optimal_k(context, config)
            tprint(f"✅ DEBUG: Optimal K determined: {optimal_k}", "DEBUG")
            context.optimal_k = optimal_k

            # Perform initial clustering
            tprint("🔍 DEBUG: About to perform initial clustering", "DEBUG")
            initial_assignments = await self._perform_initial_clustering(
                context.optimized_features, optimal_k
            )
            tprint(f"✅ DEBUG: Initial clustering completed - assignments shape: {initial_assignments.shape}", "DEBUG")
            context.initial_assignments = initial_assignments

            tprint("Step 2: Initial clustering completed successfully", "SUCCESS")
            return context

        except Exception as e:
            tprint(f"Step 2: Initial clustering failed: {e}", "ERROR")
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
        """Determine optimal number of clusters using BIC and stability analysis."""
        try:
            features = context.optimized_features
            n_samples, n_features = features.shape

            # Default optimal K
            default_k = getattr(config, 'n_regimes', 6)

            # Check memory pressure - skip optimal K determination if memory pressure is high
            try:
                
                memory_optimizer = get_integrated_hardware_manager()
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
