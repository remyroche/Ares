"""
Refactored Label Fusion Service for NAS/TAS regime analysis.

This module provides a cleaner, more maintainable implementation of the
label fusion service with better separation of concerns and error handling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import time

from sklearn.mixture import GaussianMixture

from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_structured
)

from src.utils.math_validation import (
    validate_finite, validate_numeric_array, safe_mean, safe_std
)

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


@dataclass
class LabelFusionResult:
    """Result of label fusion operation."""
    assignments: np.ndarray
    metadata: Dict[str, Any]
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'assignments': self.assignments.tolist(),
            'metadata': self.metadata,
            'execution_time': self.execution_time
        }


class LabelMappingService:
    """Service for mapping labels to shared K-space."""
    
    def __init__(self, logger: Optional[Callable] = None):
        """Initialize label mapping service."""
        self.logger = logger or tprint_info
    
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
                self.logger("Labels already aligned with target space")
                return tas_assignments, nas_assignments, {"mapping_needed": False}
            
            # Perform mapping
            if features is not None:
                return self._map_using_gmm(tas_assignments, nas_assignments, target_k, features)
            else:
                return self._create_abstain_mapping(tas_assignments, nas_assignments, target_k)
                
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
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], Dict[str, Any]]:
        """Map labels using GMM with centroid analysis."""
        try:
            # Validate features
            features = validate_finite(features, "gmm_features")
            
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
            
            mapping_info = {
                "mapping_needed": True,
                "method": "gmm_centroid",
                "tas_mapping": tas_mapping,
                "nas_mapping": nas_mapping,
            }
            
            self.logger(f"GMM mapping completed: {len(tas_mapping)} TAS, {len(nas_mapping)} NAS mappings")
            return tas_mapped, nas_mapped, tas_mapping, nas_mapping, mapping_info
            
        except Exception as e:
            tprint_error(f"GMM mapping failed: {e}")
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
            label_features = validate_finite(label_features, "label_features")
            
            # Calculate distances to centroids
            distances = np.linalg.norm(label_features[:, np.newaxis] - centroids, axis=2)
            distances = validate_finite(distances, "distances")
            
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
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], int, Dict[str, Any]]:
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
        
        mapping_info = {
            "mapping_needed": True,
            "method": "abstain_column",
            "abstain_value": abstain_value,
        }
        
        self.logger(f"Abstain mapping applied: {len(set(tas_mapped))} TAS, {len(set(nas_mapped))} NAS unique")
        return tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value, mapping_info


class DawidSkeneService:
    """Service for Dawid-Skene EM algorithm."""
    
    def __init__(self, logger: Optional[Callable] = None):
        """Initialize Dawid-Skene service."""
        self.logger = logger or tprint_info
    
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
        start_time = time.time()
        
        try:
            # Initialize mapping service
            mapping_service = LabelMappingService(self.logger)
            
            # Map labels to K-space
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
                    self.logger(f"Dawid-Skene converged after {iteration + 1} iterations")
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
            
            result = LabelFusionResult(
                assignments=fused_assignments,
                metadata=metadata,
                execution_time=time.time() - start_time
            )
            
            self.logger(f"Dawid-Skene fusion completed: {n_samples} samples, {n_classes} classes")
            return result
            
        except Exception as e:
            tprint_error(f"Dawid-Skene algorithm failed: {e}")
            raise
    
    def _initialize_confusion_matrices(self, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize confusion matrices with random values."""
        rng = np.random.default_rng(42)
        
        tas_confusion = rng.dirichlet([0.5] * n_classes, size=n_classes)
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
        try:
            n_samples, n_classes = posteriors.shape
            abstain_value = mapping_info.get("abstain_value")
            
            # Calculate posteriors for each sample
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
                    
                    # Calculate posterior
                    prior = class_priors[true_class]
                    posteriors[i, true_class] = prior * tas_factor * nas_factor
            
            # Normalize posteriors
            row_sums = posteriors.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
            posteriors[:] = posteriors / row_sums
            
        except Exception as e:
            tprint_error(f"E-step calculation failed: {e}")
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


class LabelFusionService:
    """Main service for label fusion operations."""
    
    def __init__(self, logger: Optional[Callable] = None):
        """Initialize label fusion service."""
        self.logger = logger or tprint_info
        self.mapping_service = LabelMappingService(logger)
        self.dawid_skene_service = DawidSkeneService(logger)
    
    def fuse_labels(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
        method: str = "dawid_skene",
        **kwargs
    ) -> LabelFusionResult:
        """Fuse labels using specified method."""
        try:
            if method == "dawid_skene":
                return self.dawid_skene_service.run_dawid_skene(
                    tas_assignments, nas_assignments, target_k, features, **kwargs
                )
            else:
                raise ValueError(f"Unknown fusion method: {method}")
                
        except Exception as e:
            tprint_error(f"Label fusion failed: {e}")
            raise