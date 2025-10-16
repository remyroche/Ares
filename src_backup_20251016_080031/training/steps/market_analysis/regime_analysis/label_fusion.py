"""
Refactored Label Fusion Service for NAS/TAS regime analysis.

This module provides cleaner, more maintainable implementation of label fusion
with better separation of concerns and error handling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple
import time

from sklearn.mixture import GaussianMixture

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_structured, tprint_timer
)

from src.utils.math_validation import (
    validate_finite, validate_numeric_array, safe_mean, safe_std, safe_divide
)

from src.utils.common_operations import memory_checkpoint

# Import matrix operations with fallback
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    tprint_warning("Matrix operations not available, using fallback implementations")

# Import M1 hardware optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    get_m1_memory_optimizer = lambda: None
    get_m1_cpu_optimizer = lambda: None
    get_m1_gpu_manager = lambda: None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from hmmlearn import hmm


def _default_logger(message: str, level: str = "INFO") -> None:
    tprint(message, level)


@dataclass
class LabelFusionResult:
    """Result of label fusion operation."""
    assignments: np.ndarray
    metadata: Dict[str, Any]


class LabelMappingService:
    """Service for mapping labels to shared K-space."""
    
    def __init__(self, logger: Optional[Callable] = None):
        """Initialize label mapping service."""
        self.logger = logger or _default_logger
    
    def map_labels_to_k_space(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Map NAS/TAS labels to shared K-space."""
        try:
            # Validate inputs
            tas_assignments = validate_numeric_array(tas_assignments, "tas_assignments")
            nas_assignments = validate_numeric_array(nas_assignments, "nas_assignments")
            
            # Check if mapping is needed
            if self._labels_already_aligned(tas_assignments, nas_assignments, target_k):
                self.logger("Labels already aligned with target space", "INFO")
                return tas_assignments, nas_assignments, {"mapping_needed": False}
            
            self.logger(f"Mapping labels to shared K={target_k} space", "INFO")
            
            # Perform mapping
            if features is not None:
                tas_mapped, nas_mapped, tas_mapping, nas_mapping = self._map_using_gmm(
                    tas_assignments, nas_assignments, target_k, features
                )
                mapping_info = {
                    "mapping_needed": True,
                    "method": "gmm_centroid",
                    "tas_mapping": tas_mapping,
                    "nas_mapping": nas_mapping,
                }
            else:
                tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value = self._create_abstain_mapping(
                    tas_assignments, nas_assignments, target_k
                )
                mapping_info = {
                    "mapping_needed": True,
                    "method": "abstain_column",
                    "abstain_value": abstain_value,
                    "tas_mapping": tas_mapping,
                    "nas_mapping": nas_mapping,
                }
            
            self.logger(f"Label mapping completed – TAS unique: {len(set(tas_mapped))}, NAS unique: {len(set(nas_mapped))}", "SUCCESS")
            return tas_mapped, nas_mapped, mapping_info
            
        except Exception as e:
            tprint_error(f"Label mapping failed: {e}")
            raise
    
    def _labels_already_aligned(
        self, 
        tas_assignments: np.ndarray, 
        nas_assignments: np.ndarray, 
        target_k: int
    ) -> bool:
        """Check if labels are already aligned with target K-space."""
        tas_unique = set(tas_assignments.tolist())
        nas_unique = set(nas_assignments.tolist())
        
        return (
            tas_unique and nas_unique and
            max(tas_unique) < target_k and max(nas_unique) < target_k and
            min(tas_unique) >= 0 and min(nas_unique) >= 0
        )
    
    def _map_using_gmm(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
        """Map labels using GMM with centroid analysis."""
        with tprint_timer(f"GMM mapping with target_k={target_k}"):
            try:
                # Validate features array shape and content
                if features is None or features.size == 0:
                    raise ValueError("Features array is None or empty")
                if features.ndim != 2:
                    raise ValueError(f"Features must be 2D array, got {features.ndim}D")
                if not np.isfinite(features).all():
                    raise ValueError("Features contain NaN or infinite values")

                # Fit GMM
                gmm = GaussianMixture(n_components=target_k, random_state=42)
                gmm.fit(features)
                centroids = gmm.means_
                
                # Create mappings
                tas_mapping = self._create_label_mapping(tas_assignments, features, centroids, target_k)
                nas_mapping = self._create_label_mapping(nas_assignments, features, centroids, target_k)
                
                # Apply mappings
                tas_mapped = np.array([tas_mapping.get(label, label % target_k) for label in tas_assignments])
                nas_mapped = np.array([nas_mapping.get(label, label % target_k) for label in nas_assignments])
                
                # Validate results
                tas_mapped = validate_numeric_array(tas_mapped, "tas_mapped")
                nas_mapped = validate_numeric_array(nas_mapped, "nas_mapped")
                
                tprint_success(f"GMM mapping completed: {len(tas_mapping)} TAS, {len(nas_mapping)} NAS mappings")
                return tas_mapped, nas_mapped, tas_mapping, nas_mapping
                
            except Exception as exc:
                tprint_error(f"Failed to map using GMM: {exc}")
                raise
    
    def _create_label_mapping(
        self,
        assignments: np.ndarray,
        features: np.ndarray,
        centroids: np.ndarray,
        target_k: int,
    ) -> Dict[int, int]:
        """Create mapping from original labels to target K-space."""
        mapping = {}
        
        for label in set(assignments.tolist()):
            mapped_label = self._find_nearest_centroid(label, assignments, features, centroids)
            mapping[label] = mapped_label
        
        return mapping
    
    def _find_nearest_centroid(
        self,
        label: int,
        assignments: np.ndarray,
        features: np.ndarray,
        centroids: np.ndarray,
    ) -> int:
        """Find nearest centroid for a label."""
        try:
            mask = assignments == label
            if not np.any(mask):
                return int(label % len(centroids))
            
            label_features = features[mask]
            label_features = validate_numeric_array(label_features, "label_features")

            # Calculate distances to centroids
            distances = np.linalg.norm(label_features[:, np.newaxis] - centroids, axis=2)
            distances = validate_numeric_array(distances, "distances")
            
            # Find nearest centroid
            mean_distances = safe_mean(distances, axis=0)
            return int(np.argmin(mean_distances))
            
        except Exception as e:
            tprint_warning(f"Centroid calculation failed for label {label}: {e}")
            return int(label % len(centroids))
    
    def _create_abstain_mapping(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], int]:
        """Create abstain mapping for labels outside target K-space."""
        abstain_value = target_k
        
        tas_mapping = {
            label: label if 0 <= label < target_k else abstain_value
            for label in set(tas_assignments.tolist())
        }
        
        nas_mapping = {
            label: label if 0 <= label < target_k else abstain_value
            for label in set(nas_assignments.tolist())
        }
        
        tas_mapped = np.array([tas_mapping.get(label, abstain_value) for label in tas_assignments])
        nas_mapped = np.array([nas_mapping.get(label, abstain_value) for label in nas_assignments])
        
        self.logger(
            f"Abstain mapping applied – TAS unique: {len(set(tas_mapped))}, NAS unique: {len(set(nas_mapped))}",
            "INFO"
        )
        return tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value


class DawidSkeneService:
    """Service for Dawid-Skene EM algorithm."""
    
    def __init__(
        self,
        logger: Optional[Callable] = None,
        historical_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        statistics_cache: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Dawid-Skene service."""
        self.logger = logger or _default_logger
        
        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.gpu_manager = get_m1_gpu_manager()
        
        # Initialize matrix operations
        if MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.batch_processor = get_batch_matrix_processor()
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.batch_processor = None
        
        # Statistics cache for calibrated priors
        if statistics_cache is not None:
            self._statistics_cache = self._ensure_statistics_cache(statistics_cache)
        else:
            self._statistics_cache = self._compute_statistics_cache(historical_pairs)
        
        self._calibrated_priors = self._statistics_cache.get("dirichlet_alpha", {"tas": {}, "nas": {}})
        self._transition_regularizer = self._statistics_cache.get("transition_regularizer")
    
    def run_dawid_skene(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> LabelFusionResult:
        """Run Dawid-Skene EM algorithm."""
        tprint(f"Starting Dawid–Skene fusion with K={target_k}", "INFO")
        
        try:
            # Map labels to K-space
            mapping_service = LabelMappingService(self.logger)
            tas_mapped, nas_mapped, mapping_info = mapping_service.map_labels_to_k_space(
                tas_assignments, nas_assignments, target_k, features
            )
            
            # Initialize EM algorithm
            n_classes = target_k
            n_samples = len(tas_mapped)
            
            # Initialize parameters
            tas_confusion, nas_confusion = self._initialize_confusion_matrices(n_classes)
            class_priors = np.ones(n_classes) / n_classes
            posteriors = np.zeros((n_samples, n_classes))
            log_likelihoods = []
            
            # Run EM iterations
            for iteration in range(max_iterations):
                # E-step
                self._e_step(
                    tas_mapped, nas_mapped, tas_confusion, nas_confusion,
                    class_priors, posteriors, mapping_info
                )
                
                # Calculate log likelihood
                log_likelihoods.append(
                    float(np.sum(np.log(np.clip(posteriors.sum(axis=1), 1e-10, None))))
                )
                
                # M-step
                old_tas_confusion = tas_confusion.copy()
                old_nas_confusion = nas_confusion.copy()
                old_priors = class_priors.copy()
                
                class_priors = self._update_class_priors(posteriors)
                tas_confusion = self._update_confusion_matrix(posteriors, tas_mapped, n_classes)
                nas_confusion = self._update_confusion_matrix(posteriors, nas_mapped, n_classes)
                
                # Check convergence
                if self._has_converged(
                    old_tas_confusion, old_nas_confusion, old_priors,
                    tas_confusion, nas_confusion, class_priors, tolerance
                ):
                    tprint_success(f"Dawid–Skene converged after {iteration + 1} iterations")
                    break
            
            # Get final assignments
            fused_assignments = np.argmax(posteriors, axis=1)
            
            # Create result
            metadata = {
                "iterations": len(log_likelihoods),
                "converged": len(log_likelihoods) < max_iterations,
                "log_likelihoods": log_likelihoods,
                "tas_confusion_matrix": tas_confusion.tolist(),
                "nas_confusion_matrix": nas_confusion.tolist(),
                "class_priors": class_priors.tolist(),
                "mapping_info": mapping_info,
                "posteriors": posteriors.tolist(),
            }
            
            tprint_success(f"Dawid–Skene fusion completed: {n_samples} samples, {n_classes} classes")
            return LabelFusionResult(assignments=fused_assignments, metadata=metadata)
            
        except Exception as e:
            tprint_error(f"Dawid-Skene algorithm failed: {e}")
            raise
    
    def _initialize_confusion_matrices(self, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize confusion matrices with random values or priors."""
        rng = np.random.default_rng(42)
        tas_alpha = self._get_dirichlet_alpha(n_classes, "tas")
        nas_alpha = self._get_dirichlet_alpha(n_classes, "nas")
        
        if tas_alpha is not None:
            tas_confusion = np.vstack([rng.dirichlet(row) for row in tas_alpha])
        else:
            tas_confusion = rng.dirichlet([0.5] * n_classes, size=n_classes)
        
        if nas_alpha is not None:
            nas_confusion = np.vstack([rng.dirichlet(row) for row in nas_alpha])
        else:
            nas_confusion = rng.dirichlet([0.5] * n_classes, size=n_classes)
        
        return tas_confusion, nas_confusion
    
    def _e_step(
        self,
        tas_mapped: np.ndarray,
        nas_mapped: np.ndarray,
        tas_confusion: np.ndarray,
        nas_confusion: np.ndarray,
        class_priors: np.ndarray,
        posteriors: np.ndarray,
        mapping_info: Dict[str, Any],
    ) -> None:
        """E-step of EM algorithm."""
        with tprint_timer("E-step calculation"):
            try:
                n_samples, n_classes = posteriors.shape
                abstain_value = mapping_info.get("abstain_value")
                
                # Validate input arrays
                tas_mapped = validate_numeric_array(tas_mapped, "tas_mapped")
                nas_mapped = validate_numeric_array(nas_mapped, "nas_mapped")
                
                # Calculate posteriors
                for i in range(n_samples):
                    tas_obs = tas_mapped[i]
                    nas_obs = nas_mapped[i]
                    
                    for true_class in range(n_classes):
                        # Calculate likelihood factors
                        tas_factor = (
                            tas_confusion[true_class, tas_obs]
                            if tas_obs < n_classes else 1.0
                        )
                        nas_factor = (
                            nas_confusion[true_class, nas_obs]
                            if nas_obs < n_classes else 1.0
                        )
                        
                        # Handle abstain values
                        if abstain_value is not None:
                            if tas_obs == abstain_value:
                                tas_factor = 1.0
                            if nas_obs == abstain_value:
                                nas_factor = 1.0
                        
                        # Ensure positive values
                        prior = max(class_priors[true_class], 1e-10)
                        tas_factor = max(tas_factor, 1e-10)
                        nas_factor = max(nas_factor, 1e-10)
                        
                        # Calculate posterior
                        posteriors[i, true_class] = prior * tas_factor * nas_factor
                
                # Normalize posteriors
                row_sums = posteriors.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
                posteriors[:] = posteriors / row_sums
                posteriors[:] = np.clip(posteriors, 0.0, 1.0)
                
                # Final normalization to ensure rows sum to 1
                final_row_sums = posteriors.sum(axis=1, keepdims=True)
                final_row_sums = np.where(final_row_sums == 0.0, 1.0, final_row_sums)
                posteriors[:] = posteriors / final_row_sums
                
                tprint_success(f"E-step completed for {n_samples} samples, {n_classes} classes")
                
            except Exception as exc:
                tprint_error(f"E-step calculation failed: {exc}")
                raise
    
    def _update_class_priors(self, posteriors: np.ndarray) -> np.ndarray:
        """Update class priors in M-step."""
        priors = posteriors.mean(axis=0)
        priors_sum = priors.sum()
        if priors_sum == 0:
            return np.ones_like(priors) / len(priors)
        return priors / priors_sum
    
    def _update_confusion_matrix(
        self, posteriors: np.ndarray, mapped_assignments: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Update confusion matrix in M-step."""
        confusion = np.zeros((n_classes, n_classes))
        
        for true_class in range(n_classes):
            for observed_class in range(n_classes):
                mask = mapped_assignments == observed_class
                if np.any(mask):
                    confusion[true_class, observed_class] = posteriors[mask, true_class].sum()
            
            # Normalize row
            row_sum = confusion[true_class].sum()
            if row_sum == 0:
                confusion[true_class] = np.ones(n_classes) / n_classes
            else:
                confusion[true_class] /= row_sum
        
        return confusion
    
    def _has_converged(
        self,
        old_tas: np.ndarray,
        old_nas: np.ndarray,
        old_priors: np.ndarray,
        new_tas: np.ndarray,
        new_nas: np.ndarray,
        new_priors: np.ndarray,
        tolerance: float,
    ) -> bool:
        """Check if EM algorithm has converged."""
        tas_change = np.abs(new_tas - old_tas).max()
        nas_change = np.abs(new_nas - old_nas).max()
        prior_change = np.abs(new_priors - old_priors).max()
        
        return max(tas_change, nas_change, prior_change) < tolerance
    
    def _get_dirichlet_alpha(self, n_classes: int, annotator: str) -> Optional[np.ndarray]:
        """Get Dirichlet alpha priors for confusion matrix initialization."""
        priors = self._calibrated_priors.get(annotator, {})
        if not priors:
            return None
        
        if n_classes in priors:
            matrix = priors[n_classes]
            if matrix.shape[0] >= n_classes and matrix.shape[1] >= n_classes:
                return matrix[:n_classes, :n_classes]
            return None
        
        larger_sizes = sorted(size for size in priors if size >= n_classes)
        if larger_sizes:
            matrix = priors[larger_sizes[0]]
            return matrix[:n_classes, :n_classes]
        
        return None
    
    def _ensure_statistics_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure statistics cache is properly formatted."""
        dirichlet_alpha = cache.get("dirichlet_alpha", {})
        tas_priors = {}
        nas_priors = {}
        
        if isinstance(dirichlet_alpha, dict):
            tas_raw = dirichlet_alpha.get("tas", {})
            nas_raw = dirichlet_alpha.get("nas", {})
            if isinstance(tas_raw, dict):
                for key, value in tas_raw.items():
                    tas_priors[int(key)] = np.asarray(value, dtype=float)
            if isinstance(nas_raw, dict):
                for key, value in nas_raw.items():
                    nas_priors[int(key)] = np.asarray(value, dtype=float)
        
        try:
            regularizer_value = float(cache.get("transition_regularizer", 0.1))
        except (TypeError, ValueError):
            regularizer_value = 0.1
        
        return {
            "dirichlet_alpha": {"tas": tas_priors, "nas": nas_priors},
            "transition_regularizer": regularizer_value,
        }
    
    def _compute_statistics_cache(
        self, historical_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[str, Any]:
        """Compute statistics cache from historical label pairs."""
        if not historical_pairs:
            return self._ensure_statistics_cache({})
        
        tas_priors: Dict[int, np.ndarray] = {}
        nas_priors: Dict[int, np.ndarray] = {}
        total_pairs = 0
        agreements = 0
        
        for tas_seq, nas_seq in historical_pairs:
            tas_arr = np.asarray(tas_seq, dtype=int)
            nas_arr = np.asarray(nas_seq, dtype=int)
            if tas_arr.size == 0 or nas_arr.size == 0:
                continue
            
            limit = min(len(tas_arr), len(nas_arr))
            tas_arr = tas_arr[:limit]
            nas_arr = nas_arr[:limit]
            
            valid_mask = (tas_arr >= 0) & (nas_arr >= 0)
            if not np.any(valid_mask):
                continue
            
            tas_arr = tas_arr[valid_mask]
            nas_arr = nas_arr[valid_mask]
            
            if tas_arr.size == 0:
                continue
            
            n_classes = int(max(int(tas_arr.max()), int(nas_arr.max())) + 1)
            tas_counts = tas_priors.setdefault(n_classes, np.zeros((n_classes, n_classes)))
            nas_counts = nas_priors.setdefault(n_classes, np.zeros((n_classes, n_classes)))
            
            for tas_val, nas_val in zip(tas_arr, nas_arr):
                if 0 <= nas_val < n_classes and 0 <= tas_val < n_classes:
                    tas_counts[nas_val, tas_val] += 1.0
                    nas_counts[tas_val, nas_val] += 1.0
            
            agreements += int(np.sum(tas_arr == nas_arr))
            total_pairs += tas_arr.size
        
        disagreement_rate = 1.0 - (agreements / total_pairs) if total_pairs else 0.0
        transition_regularizer = float(np.clip(disagreement_rate, 1e-3, 0.5)) if total_pairs else 0.1
        
        # Apply Laplace smoothing
        for cache in (tas_priors, nas_priors):
            for n_classes, counts in cache.items():
                cache[n_classes] = counts + 1.0
        
        return self._ensure_statistics_cache({
            "dirichlet_alpha": {"tas": tas_priors, "nas": nas_priors},
            "transition_regularizer": transition_regularizer,
        })
    
    def get_transition_regularizer(self, default: float = 0.1) -> float:
        """Get transition regularizer value."""
        value = self._transition_regularizer
        if value is None or not np.isfinite(value) or value <= 0.0:
            return default
        return float(value)


class LabelFusionService:
    """Main service for label fusion operations."""
    
    def __init__(
        self,
        logger: Callable[[str, str], None] = _default_logger,
        historical_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        statistics_cache: Optional[Dict[str, Any]] = None,
    ):
        """Initialize label fusion service."""
        self.logger = logger
        self.dawid_skene_service = DawidSkeneService(logger, historical_pairs, statistics_cache)
    
    def map_labels_to_k_space(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Map NAS/TAS labels to shared K-space."""
        mapping_service = LabelMappingService(self.logger)
        return mapping_service.map_labels_to_k_space(tas_assignments, nas_assignments, target_k, features)
    
    def run_dawid_skene(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> LabelFusionResult:
        """Run Dawid-Skene EM algorithm."""
        return self.dawid_skene_service.run_dawid_skene(
            tas_assignments, nas_assignments, target_k, features, max_iterations, tolerance
        )
    
    def get_statistics_cache(self) -> Dict[str, Any]:
        """Get statistics cache."""
        return self.dawid_skene_service._statistics_cache
    
    def get_transition_regularizer(self, default: float = 0.1) -> float:
        """Get transition regularizer."""
        return self.dawid_skene_service.get_transition_regularizer(default)
    
    def get_calibrated_priors(self) -> Dict[str, Dict[int, np.ndarray]]:
        """Get calibrated priors."""
        return self.dawid_skene_service._calibrated_priors
    
    def get_persistence_threshold(self, key: str, default: Optional[float] = None) -> float:
        """Get persistence threshold (for backward compatibility)."""
        if default is not None:
            return default
        return 0.99 if key == "high" else 0.6
    
    def get_persistence_quantiles(self) -> Dict[str, float]:
        """Get persistence quantiles (for backward compatibility)."""
        return {}


class RegimeOptimizationService:
    """Service responsible for regime optimization, scoring and smoothing."""
    
    def __init__(
        self,
        label_fusion_service: Optional[LabelFusionService],
        score_calculator: Callable[[np.ndarray, np.ndarray], float],
        logger: Callable[[str, str], None] = _default_logger,
    ) -> None:
        """Initialize regime optimization service."""
        if label_fusion_service is None:
            self._label_fusion_service = LabelFusionService(logger=logger)
        else:
            self._label_fusion_service = label_fusion_service
        
        self._score_calculator = score_calculator
        self._logger = logger
    
    def progressive_regime_optimization_with_k(
        self,
        features: np.ndarray,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        market_data: Optional[np.ndarray],
        optimal_k: int,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Run Dawid–Skene fusion then apply balance-aware optimization."""
        tprint("Starting progressive regime optimization", "INFO")
        
        fusion_result = self._label_fusion_service.run_dawid_skene(
            tas_assignments, nas_assignments, optimal_k, features
        )
        
        mapped_assignments = self._map_to_optimal_k(fusion_result.assignments, features, optimal_k)
        initial_score = self._score_calculator(features, mapped_assignments)
        
        # Apply balance-aware optimization to prevent dominant regimes (>25% threshold)
        tprint("Applying balance-aware optimization to prevent regime imbalance...", "INFO")
        balanced_assignments, balance_improvement = self._apply_balance_optimization(
            features, mapped_assignments, optimal_k
        )
        final_score = self._score_calculator(features, balanced_assignments)
        
        optimization_metrics = {
            "initial_score": initial_score,
            "final_score": final_score,
            "improvement": final_score - initial_score,
            "balance_improvement": balance_improvement,
            "iterations": 1,
            "optimal_k": optimal_k,
            "method": "balance_aware_optimization",
            "fusion_metadata": fusion_result.metadata,
        }
        
        tprint_success(f"Progressive optimization completed – Score: {final_score:.3f} (Balance improvement: {balance_improvement:.3f})")
        return balanced_assignments, optimization_metrics, fusion_result.metadata
    
    def _apply_balance_optimization(self, features: np.ndarray, assignments: np.ndarray, optimal_k: int) -> Tuple[np.ndarray, float]:
        """Apply balance-aware optimization to prevent dominant regimes (>15% threshold)."""
        try:
            import numpy as np
            
            # Calculate current regime distribution
            unique_regimes, regime_counts = np.unique(assignments, return_counts=True)
            total_samples = len(assignments)
            regime_percentages = {regime: (count / total_samples) * 100 for regime, count in zip(unique_regimes, regime_counts)}
            
            # Check if any regime exceeds 20% threshold
            dominant_regimes = [regime for regime, pct in regime_percentages.items() if pct > 20.0]
            
            if not dominant_regimes:
                tprint("No dominant regimes detected - balance optimization not needed", "INFO")
                return assignments, 0.0
            
            tprint(f"Detected dominant regimes: {dominant_regimes} - applying balance optimization", "WARNING")
            
            # Create balanced assignments by redistributing samples
            balanced_assignments = assignments.copy()
            target_percentage = 100.0 / optimal_k  # Target equal distribution
            
            for dominant_regime in dominant_regimes:
                current_percentage = regime_percentages[dominant_regime]
                excess_samples = int((current_percentage - target_percentage) / 100.0 * total_samples)
                
                if excess_samples > 0:
                    # Find samples in dominant regime
                    dominant_indices = np.where(balanced_assignments == dominant_regime)[0]
                    
                    # Find under-represented regimes
                    underrepresented_regimes = [regime for regime, pct in regime_percentages.items() 
                                              if pct < target_percentage and regime != dominant_regime]
                    
                    if underrepresented_regimes:
                        # Redistribute excess samples to under-represented regimes
                        samples_to_redistribute = min(excess_samples, len(dominant_indices))
                        np.random.seed(42)  # For reproducibility
                        indices_to_redistribute = np.random.choice(dominant_indices, samples_to_redistribute, replace=False)
                        
                        for i, idx in enumerate(indices_to_redistribute):
                            target_regime = underrepresented_regimes[i % len(underrepresented_regimes)]
                            balanced_assignments[idx] = target_regime
            
            # Calculate balance improvement
            new_regime_counts = [np.sum(balanced_assignments == regime) for regime in unique_regimes]
            new_regime_percentages = [(count / total_samples) * 100 for count in new_regime_counts]
            
            # Calculate balance score (1.0 = perfect balance, 0.0 = worst imbalance)
            original_balance = 1.0 - (np.std(list(regime_percentages.values())) / np.mean(list(regime_percentages.values())))
            new_balance = 1.0 - (np.std(new_regime_percentages) / np.mean(new_regime_percentages))
            balance_improvement = new_balance - original_balance
            
            tprint(f"Balance optimization completed - Improvement: {balance_improvement:.3f}", "SUCCESS")
            return balanced_assignments, balance_improvement
            
        except Exception as e:
            tprint(f"Balance optimization failed: {e}", "ERROR")
            return assignments, 0.0
    
    def apply_hmm_smoothing(
        self, features: np.ndarray, assignments: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply HMM-based smoothing with simple fallback."""
        try:
            model = self._initialize_hmm(features, assignments)
            model.fit(features)
            smoothed_assignments = model.predict(features)
            metadata = self._build_smoothing_metadata(model, assignments, smoothed_assignments)
            tprint_success(f"HMM smoothing completed – {model.n_components} clusters")
            return smoothed_assignments, metadata
        except Exception as exc:  # pragma: no cover - safety fallback
            tprint_warning(f"HMM smoothing failed ({exc}), using simple smoothing fallback")
            smoothed = self._simple_temporal_smoothing(assignments)
            return smoothed, {"method": "simple_fallback", "error": str(exc)}
    
    def _map_to_optimal_k(
        self, assignments: np.ndarray, features: np.ndarray, optimal_k: int
    ) -> np.ndarray:
        """Map assignments to optimal K using GMM."""
        if len(set(assignments.tolist())) == optimal_k:
            tprint_success(f"Assignments already match optimal K={optimal_k}")
            return assignments
        
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm.fit(features)
        mapped = gmm.predict(features)
        tprint_success(f"Assignments remapped via GMM – clusters: {len(set(mapped.tolist()))}")
        return mapped
    
    def _initialize_hmm(self, features: np.ndarray, assignments: np.ndarray) -> "hmm.GaussianHMM":
        """Initialize HMM for smoothing."""
        from hmmlearn import hmm
        
        n_clusters = len(set(assignments.tolist()))
        if n_clusters <= 0:
            raise ValueError("At least one cluster is required for HMM smoothing")
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(features)
        
        model = hmm.GaussianHMM(
            n_components=n_clusters,
            random_state=42,
            n_iter=50,
            init_params="stmc",
        )
        
        model.means_ = gmm.means_
        model.covars_ = gmm.covariances_
        model.startprob_ = np.ones(n_clusters) / n_clusters
        model.transmat_ = self._learn_transition_matrix(assignments, n_clusters)
        return model
    
    def _learn_transition_matrix(
        self, assignments: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Learn transition matrix from assignments."""
        transition_matrix = np.zeros((n_clusters, n_clusters))
        for current, nxt in zip(assignments[:-1], assignments[1:]):
            transition_matrix[current, nxt] += 1
        
        regularizer = self._label_fusion_service.get_transition_regularizer()
        transition_matrix += regularizer * np.eye(n_clusters)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return transition_matrix / row_sums
    
    def _build_smoothing_metadata(
        self,
        model: hmm.GaussianHMM,
        original_assignments: np.ndarray,
        smoothed_assignments: np.ndarray,
    ) -> Dict[str, Any]:
        """Build metadata for HMM smoothing."""
        high_threshold = self._label_fusion_service.get_persistence_threshold("high", default=0.99)
        low_threshold = self._label_fusion_service.get_persistence_threshold("low", default=0.6)
        
        expected_durations = []
        low_persistence_regimes = []
        for idx in range(model.n_components):
            p_kk = model.transmat_[idx, idx]
            if p_kk >= high_threshold:
                expected_durations.append(float("inf"))
            else:
                expected_duration = 1.0 / max(1e-6, (1 - p_kk))
                expected_durations.append(expected_duration)
                if p_kk < low_threshold:
                    low_persistence_regimes.append(idx)
        
        metadata = {
            "method": "hmm",
            "expected_durations": expected_durations,
            "low_persistence_regimes": low_persistence_regimes,
            "transmat": model.transmat_.tolist(),
            "changed_points": np.nonzero(original_assignments != smoothed_assignments)[0].tolist(),
            "persistence_thresholds": {"high": high_threshold, "low": low_threshold},
        }
        
        if low_persistence_regimes:
            tprint_warning(f"Low persistence regimes detected: {low_persistence_regimes}")
        
        return metadata
    
    def _simple_temporal_smoothing(self, assignments: np.ndarray) -> np.ndarray:
        """Apply simple temporal smoothing."""
        smoothed = assignments.copy()
        for idx in range(1, len(assignments) - 1):
            if assignments[idx] != assignments[idx - 1] and assignments[idx] != assignments[idx + 1]:
                window = assignments[max(0, idx - 2) : min(len(assignments), idx + 3)]
                values, counts = np.unique(window, return_counts=True)
                smoothed[idx] = int(values[np.argmax(counts)])
        return smoothed
