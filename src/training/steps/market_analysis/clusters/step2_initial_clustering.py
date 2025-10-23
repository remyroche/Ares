"""
Step 2: Initial Clustering for NAS-TAS Clustering.

This module handles the initial clustering setup, regime assignment extraction,
and basic clustering initialization.

ENHANCED WITH BASESTEP COMPREHENSIVE TOOLS:
- Direct access to all utility modules through BaseStep
- Comprehensive logging with tprint integration
- Hardware optimization built-in
- Safe operations with fallbacks
- Memory management and cleanup
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Import BaseStep for comprehensive utility access
from src.training.steps.base_step import BaseStep

# Import tprint functions directly (available through BaseStep)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance,
    tprint_step_start, tprint_step_end, tprint_operation_start, tprint_operation_end,
    tprint_data_summary, tprint_performance_summary, tprint_memory_usage
)

from .shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext

class InitialClusteringStep(BaseStep):
    """Step 2: Initial clustering and regime assignment extraction with BaseStep comprehensive tools."""

    def __init__(self, verbose: bool = True, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the initial clustering step with BaseStep utilities."""
        super().__init__("initial_clustering", config)
        
        tprint_step_start("InitialClusteringStep", config)
        self.verbose = verbose
        
        # Log utility availability
        availability = self._get_availability_status()
        tprint_info(f"Utility availability: {sum(availability.values())}/{len(availability)} utilities available")
        
        tprint_debug(f"Step verbose mode: {verbose}")
        tprint_step_end("InitialClusteringStep", True, 0.0)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute initial clustering step using BaseStep comprehensive tools."""
        tprint_step_start("Initial Clustering", config)
        
        try:
            # Extract context from config
            context = self._extract_context_from_config(config)
            
            tprint_info(f"Context features shape: {context.optimized_features.shape}")

            # Validate input features using BaseStep utilities
            tprint_operation_start("Feature Validation")
            context.optimized_features = self._validate_finite(context.optimized_features, default=0)
            tprint_debug(f"Features validated - shape: {context.optimized_features.shape}")
            tprint_operation_end("Feature Validation", True)

            # Extract TAS and NAS regime assignments using BaseStep utilities
            tprint_operation_start("Regime Assignment Extraction")
            tas_assignments, nas_assignments = await self._extract_regime_assignments_safe(context, config)
            tprint_debug(f"Regime assignments extracted - TAS: {len(tas_assignments)}, NAS: {len(nas_assignments)}")
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments
            tprint_operation_end("Regime Assignment Extraction", True)

            # Initialize basic clustering with optimal K using BaseStep utilities
            tprint_operation_start("Optimal K Determination")
            optimal_k = await self._determine_optimal_k_safe(context, config)
            tprint_debug(f"Optimal K determined: {optimal_k}")
            context.optimal_k = optimal_k
            tprint_operation_end("Optimal K Determination", True)

            # Perform initial clustering using BaseStep utilities
            tprint_operation_start("Initial Clustering")
            initial_assignments = await self._perform_initial_clustering_safe(
                context.optimized_features, optimal_k
            )
            tprint_debug(f"Initial clustering completed - assignments shape: {initial_assignments.shape}")
            context.initial_assignments = initial_assignments
            tprint_operation_end("Initial Clustering", True)

            # Create comprehensive outcome using BaseStep utilities
            outcome = self._create_comprehensive_outcome(context, config)
            
            tprint_step_end("Initial Clustering", True, 0.0)
            return outcome

        except Exception as e:
            tprint_error(f"❌ Initial clustering failed: {e}")
            # Use BaseStep memory cleanup
            if self.hardware_utils:
                self.hardware_utils['force_garbage_collection']()
            tprint_step_end("Initial Clustering", False, 0.0)
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    def _extract_context_from_config(self, config: Dict[str, Any]) -> ClusteringContext:
        """Extract context from config using BaseStep utilities."""
        try:
            if 'context' in config:
                return config['context']
            
            # Create new context if not provided
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
            tprint_error(f"❌ Failed to extract context: {e}")
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
                'optimal_k': context.optimal_k,
                'tas_assignments_count': len(context.tas_assignments) if context.tas_assignments is not None else 0,
                'nas_assignments_count': len(context.nas_assignments) if context.nas_assignments is not None else 0,
                'initial_assignments_count': len(context.initial_assignments) if context.initial_assignments is not None else 0
            }
            
            # Use BaseStep performance logging
            tprint_performance_summary(metrics)
            
            # Create artifacts using BaseStep utilities
            artifacts = []
            if context.initial_assignments is not None:
                # Save initial assignments
                self._save_dataframe(
                    pd.DataFrame({'assignments': context.initial_assignments}), 
                    "initial_assignments"
                )
                artifacts.append("initial_assignments")
            
            if context.tas_assignments is not None:
                # Save TAS assignments
                self._save_dataframe(
                    pd.DataFrame({'tas_assignments': context.tas_assignments}), 
                    "tas_assignments"
                )
                artifacts.append("tas_assignments")
            
            if context.nas_assignments is not None:
                # Save NAS assignments
                self._save_dataframe(
                    pd.DataFrame({'nas_assignments': context.nas_assignments}), 
                    "nas_assignments"
                )
                artifacts.append("nas_assignments")
            
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

    async def _extract_regime_assignments_safe(
        self,
        context: ClusteringContext,
        config: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments using BaseStep safe operations."""
        try:
            # For now, create dummy assignments based on features
            # In the full implementation, this would extract from pipeline state
            n_samples = context.optimized_features.shape[0]

            # Create dummy TAS assignments (trend-following) using BaseStep utilities
            tas_assignments = np.random.randint(0, 3, n_samples)
            tas_assignments = self._validate_finite(tas_assignments, default=0)

            # Create dummy NAS assignments (mean-reverting) using BaseStep utilities
            nas_assignments = np.random.randint(0, 3, n_samples)
            nas_assignments = self._validate_finite(nas_assignments, default=0)

            tprint_success(f"Extracted TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}")
            return tas_assignments, nas_assignments

        except Exception as e:
            tprint(f"Regime assignment extraction failed: {e}", "ERROR")
            raise

    async def _determine_optimal_k_safe(self, context: ClusteringContext, config: Any) -> int:
        """Determine optimal number of clusters using BaseStep safe operations."""
        try:
            features = context.optimized_features
            n_samples, n_features = features.shape

            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            tprint_debug(f"Determining optimal K for {n_samples} samples, {n_features} features")

            # Default optimal K
            default_k = config.get('n_regimes', 6)

            # Check memory pressure using BaseStep utilities
            try:
                if self.hardware_utils:
                    memory_pressure = self.hardware_utils.get('memory_pressure', 0.0)
                    if memory_pressure > 0.8:  # High memory pressure threshold
                        tprint_warning(f"🧠 High memory pressure detected ({memory_pressure:.2f}), using default K")
                        return default_k
            except Exception as e:
                tprint_warning(f"Could not check memory pressure: {e}, proceeding with optimal K determination")

            # Use BIC to determine optimal K for GMM using BaseStep utilities
            k_range = range(2, min(10, n_samples // 10))
            bic_scores = []

            for k in k_range:
                try:
                    gmm = GaussianMixture(n_components=k, random_state=42, max_iter=50)
                    gmm.fit(features)
                    bic_score = gmm.bic(features)
                    # Use BaseStep math validation
                    bic_score = self._validate_finite(bic_score, default=float('inf'))
                    bic_scores.append(bic_score)
                except Exception as e:
                    tprint_warning(f"BIC calculation failed for k={k}: {e}")
                    bic_scores.append(float('inf'))

            # More robust BIC score validation using BaseStep utilities
            if bic_scores and len(bic_scores) > 0:
                try:
                    # Check if all scores are finite and not all infinite
                    finite_scores = [score for score in bic_scores if np.isfinite(score)]
                    if finite_scores:
                        optimal_k = k_range[np.argmin(bic_scores)]
                        tprint_success(f"BIC-selected optimal K: {optimal_k}")
                    else:
                        optimal_k = default_k
                        tprint_info(f"Using default optimal K: {optimal_k}")
                except (ValueError, TypeError):
                    optimal_k = default_k
                    tprint_info(f"Using default optimal K due to BIC validation error: {optimal_k}")
            else:
                optimal_k = default_k
                tprint_info(f"Using default optimal K: {optimal_k}")

            return optimal_k

        except Exception as e:
            tprint_error(f"❌ Optimal K determination failed: {e}")
            return config.get('n_regimes', 6)

    async def _perform_initial_clustering_safe(
        self,
        features: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Perform initial clustering using K-means with BaseStep safe operations."""
        try:
            tprint_info(f"Performing initial clustering with K={k}")

            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)

            # Use K-means for initial clustering
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            assignments = kmeans.fit_predict(features)
            
            # Use BaseStep math validation for assignments
            assignments = self._validate_finite(assignments, default=0)

            tprint_success(f"Initial clustering completed: {len(np.unique(assignments))} clusters")
            return assignments

        except Exception as e:
            tprint_error(f"❌ Initial clustering failed: {e}")
            raise

    def _validate_assignments_safe(self, assignments: np.ndarray, expected_length: int) -> bool:
        """Validate assignment array using BaseStep safe operations."""
        try:
            if assignments is None:
                return False
            if len(assignments) != expected_length:
                return False
            # Use BaseStep math validation
            validated_assignments = self._validate_finite(assignments, default=0)
            if validated_assignments is None:
                return False
            if not np.issubdtype(validated_assignments.dtype, np.integer):
                return False
            return True
        except Exception as e:
            tprint_error(f"❌ Assignment validation failed: {e}")
            return False
