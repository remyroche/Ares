"""Services for NAS/TAS label fusion and regime optimization utilities with enhanced matrix operations and M1 optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
from sklearn.mixture import GaussianMixture

# Import common operations for data quality and validation
from src.utils.common_operations import (
    validate_dataframe_columns,
    calculate_data_quality_metrics,
    create_data_quality_report,
    safe_convert_dtypes,
    optimize_dataframe_dtypes,
    get_dataframe_info,
    create_summary_statistics,
    safe_fillna,
    safe_merge_dataframes,
    safe_drop_columns,
    safe_rename_columns,
    validate_timestamp_column,
    safe_timestamp_conversion,
    safe_resample,
    align_dataframes,
    validate_dataframe_schema,
    guard_dataframe_nulls,
    get_memory_usage,
    optimize_memory,
    memory_checkpoint,
    gpu_context
)

# Import math validation for safe operations
from src.utils.math_validation import (
    safe_mean,
    safe_std,
    safe_correlation,
    safe_covariance,
    validate_finite,
    safe_percentage_change,
    safe_weighted_average,
    safe_kelly_calculation,
    safe_percentile,
    safe_matrix_inverse,
    validate_correlation_matrix,
    validate_numeric_array,
    safe_log,
    safe_sqrt,
    safe_power
)

# Import matrix operations for optimization
try:
    from src.utils.matrix_operations import (
        safe_matrix_multiply,
        batch_matrix_multiply,
        optimize_matrix_operation_with_hardware,
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_correlation_matrix,
        gpu_matrix_multiply,
        correlation_matrix_gpu,
        optimize_dataframe,
        vectorized_rolling_features,
        matrix_correlation_analysis,
        batch_matrix_multiply,
        batch_feature_transformation,
        batch_correlation_analysis,
        get_hardware_performance_report,
        cleanup_hardware_resources,
        get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    # Fast fail without fallback - matrix operations are critical
    def safe_matrix_multiply(a, b): 
        raise ImportError("Matrix operations not available - cannot proceed without hardware optimizations")
    def batch_matrix_multiply(matrices): 
        raise ImportError("Matrix operations not available - cannot proceed without hardware optimizations")
    def optimize_matrix_operation_with_hardware(op_name): 
        raise ImportError("Matrix operations not available - cannot proceed without hardware optimizations")

# Import tprint for enhanced logging
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_performance,
    tprint_timer,
    tprint_structured
)

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
    assignments: np.ndarray
    metadata: Dict[str, Any]


class LabelFusionService:
    """Service responsible for aligning NAS/TAS labels and running Dawid–Skene with enhanced matrix operations and M1 optimizations."""

    def __init__(
        self,
        logger: Callable[[str, str], None] = _default_logger,
        historical_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        statistics_cache: Optional[Dict[str, Any]] = None,
    ):
        self._logger = logger

        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.gpu_manager = get_m1_gpu_manager()

        # Initialize matrix operations with fast fail
        if MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.batch_processor = get_batch_matrix_processor()
        else:
            raise ImportError("Matrix operations are required but not available - cannot initialize LabelFusionService")

        # Statistics cache for calibrated priors and persistence thresholds
        if statistics_cache is not None:
            self._statistics_cache = self._ensure_statistics_cache(statistics_cache)
        else:
            self._statistics_cache = self._compute_statistics_cache(historical_pairs)

        self._calibrated_priors = self._statistics_cache.get("dirichlet_alpha", {"tas": {}, "nas": {}})
        self._transition_regularizer = self._statistics_cache.get("transition_regularizer")
        self._persistence_thresholds = self._statistics_cache.get("persistence_thresholds", {})
        self._persistence_quantiles = self._statistics_cache.get("persistence_quantiles", {})

        tprint_info("LabelFusionService initialized with hardware optimizations")

    def map_labels_to_k_space(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Map NAS/TAS labels to the shared ``0..K-1`` space when possible."""

        tas_unique = set(tas_assignments.tolist())
        nas_unique = set(nas_assignments.tolist())

        if (
            tas_unique
            and nas_unique
            and max(tas_unique) < target_k
            and max(nas_unique) < target_k
            and min(tas_unique) >= 0
            and min(nas_unique) >= 0
        ):
            tprint("Labels already aligned with target space", "INFO")
            return tas_assignments, nas_assignments, {"mapping_needed": False}

        tprint(
            f"Mapping labels to shared K={target_k} space (TAS={len(tas_unique)}, NAS={len(nas_unique)})",
            "INFO"
        )

        if features is not None:
            tas_mapped, nas_mapped, tas_mapping, nas_mapping = self._map_using_gmm(
                tas_assignments, nas_assignments, target_k, features
            )
            method = "gmm_centroid"
            mapping_details: Dict[str, Any] = {}
        else:
            tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value = self._create_abstain_mapping(
                tas_assignments, nas_assignments, target_k
            )
            method = "abstain_column"
            mapping_details = {"abstain_value": abstain_value}

        mapping_info = {
            "mapping_needed": True,
            "tas_mapping": tas_mapping,
            "nas_mapping": nas_mapping,
            "method": method,
            **mapping_details,
        }

        tprint_success(
            f"Label mapping completed – TAS unique: {len(set(tas_mapped))}, NAS unique: {len(set(nas_mapped))}"
        )
        return tas_mapped, nas_mapped, mapping_info

    def run_dawid_skene(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> LabelFusionResult:
        """Run Dawid–Skene EM to fuse NAS and TAS labels."""

        tprint(f"Starting Dawid–Skene fusion with K={target_k}", "INFO")

        (
            tas_mapped,
            nas_mapped,
            mapping_info,
        ) = self.map_labels_to_k_space(tas_assignments, nas_assignments, target_k, features)

        n_classes = target_k
        n_samples = len(tas_mapped)

        tas_confusion, nas_confusion = self._initialize_confusion_matrices(n_classes)
        class_priors = np.ones(n_classes) / n_classes

        posteriors = np.zeros((n_samples, n_classes))
        log_likelihoods = []

        for iteration in range(max_iterations):
            self._e_step(
                tas_mapped,
                nas_mapped,
                tas_confusion,
                nas_confusion,
                class_priors,
                posteriors,
                mapping_info,
            )

            log_likelihoods.append(
                float(np.sum(np.log(np.clip(posteriors.sum(axis=1), 1e-10, None))))
            )

            old_tas_confusion = tas_confusion.copy()
            old_nas_confusion = nas_confusion.copy()
            old_priors = class_priors.copy()

            class_priors = self._update_class_priors(posteriors)
            tas_confusion = self._update_confusion_matrix(posteriors, tas_mapped, n_classes)
            nas_confusion = self._update_confusion_matrix(posteriors, nas_mapped, n_classes)

            if self._has_converged(
                old_tas_confusion,
                old_nas_confusion,
                old_priors,
                tas_confusion,
                nas_confusion,
                class_priors,
                tolerance,
            ):
                tprint_success(f"Dawid–Skene converged after {iteration + 1} iterations")
                break

        fused_assignments = np.argmax(posteriors, axis=1)

        metadata: Dict[str, Any] = {
            "iterations": len(log_likelihoods),
            "converged": len(log_likelihoods) < max_iterations,
            "log_likelihoods": log_likelihoods,
            "tas_confusion_matrix": tas_confusion.tolist(),
            "nas_confusion_matrix": nas_confusion.tolist(),
            "class_priors": class_priors.tolist(),
            "mapping_info": mapping_info,
            "posteriors": posteriors.tolist(),
            "tas_row_sums": tas_confusion.sum(axis=1).tolist(),
            "nas_row_sums": nas_confusion.sum(axis=1).tolist(),
        }

        tprint_success(
            f"Dawid–Skene fusion completed: {n_samples} samples, {n_classes} classes"
        )

        return LabelFusionResult(assignments=fused_assignments, metadata=metadata)

    def get_statistics_cache(self) -> Dict[str, Any]:
        return self._statistics_cache

    def get_calibrated_priors(self) -> Dict[str, Dict[int, np.ndarray]]:
        return self._calibrated_priors

    def get_transition_regularizer(self, default: float = 0.1) -> float:
        value = self._transition_regularizer
        if value is None or not np.isfinite(value) or value <= 0.0:
            return default
        return float(value)

    def get_persistence_threshold(self, key: str, default: Optional[float] = None) -> float:
        threshold = self._persistence_thresholds.get(key)
        if threshold is None:
            if default is not None:
                return default
            return 0.99 if key == "high" else 0.6
        return float(threshold)

    def get_persistence_quantiles(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in self._persistence_quantiles.items()}

    def _ensure_statistics_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
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

        persistence_thresholds = cache.get("persistence_thresholds", {})
        persistence_quantiles = cache.get("persistence_quantiles", {})

        normalized_thresholds: Dict[str, float] = {}
        for key, value in persistence_thresholds.items():
            threshold_key = str(key) if isinstance(key, float) else key
            try:
                normalized_thresholds[threshold_key] = float(value)
            except (TypeError, ValueError):
                continue

        try:
            regularizer_value = float(cache.get("transition_regularizer"))
        except (TypeError, ValueError):
            regularizer_value = None

        normalized_quantiles: Dict[str, float] = {}
        if isinstance(persistence_quantiles, dict):
            for key, value in persistence_quantiles.items():
                try:
                    normalized_quantiles[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        try:
            disagreement_rate_value = (
                float(cache.get("disagreement_rate"))
                if cache.get("disagreement_rate") is not None
                else None
            )
        except (TypeError, ValueError):
            disagreement_rate_value = None

        normalized_cache = {
            "dirichlet_alpha": {"tas": tas_priors, "nas": nas_priors},
            "transition_regularizer": regularizer_value,
            "persistence_thresholds": normalized_thresholds,
            "persistence_quantiles": normalized_quantiles,
            "dwell_times": cache.get("dwell_times", []),
            "disagreement_rate": disagreement_rate_value,
        }

        # Ensure thresholds have sensible defaults if missing
        if "high" not in normalized_cache["persistence_thresholds"]:
            normalized_cache["persistence_thresholds"]["high"] = 0.99
        if "low" not in normalized_cache["persistence_thresholds"]:
            normalized_cache["persistence_thresholds"]["low"] = 0.6

        return normalized_cache

    def _compute_statistics_cache(
        self, historical_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[str, Any]:
        if not historical_pairs:
            return self._ensure_statistics_cache({})

        tas_priors: Dict[int, np.ndarray] = {}
        nas_priors: Dict[int, np.ndarray] = {}
        dwell_times: List[float] = []
        self_transition_probs: List[float] = []
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

            for sequence in (tas_arr, nas_arr):
                if sequence.size < 2:
                    continue

                # Dwell times from consecutive identical values
                run_length = 1
                for idx in range(1, sequence.size):
                    if sequence[idx] == sequence[idx - 1]:
                        run_length += 1
                    else:
                        dwell_times.append(float(run_length))
                        run_length = 1
                dwell_times.append(float(run_length))

                transition_counts = np.zeros((n_classes, n_classes))
                for current, nxt in zip(sequence[:-1], sequence[1:]):
                    if 0 <= current < n_classes and 0 <= nxt < n_classes:
                        transition_counts[current, nxt] += 1.0

                for state in range(n_classes):
                    row_sum = transition_counts[state].sum()
                    if row_sum > 0:
                        self_transition_probs.append(float(transition_counts[state, state] / row_sum))

        disagreement_rate = 1.0 - (agreements / total_pairs) if total_pairs else 0.0
        transition_regularizer = float(np.clip(disagreement_rate, 1e-3, 0.5)) if total_pairs else 0.1

        for cache in (tas_priors, nas_priors):
            for n_classes, counts in cache.items():
                cache[n_classes] = counts + 1.0  # Laplace smoothing to keep Dirichlet positive

        if self_transition_probs:
            quantiles = {
                0.25: float(np.quantile(self_transition_probs, 0.25)),
                0.5: float(np.quantile(self_transition_probs, 0.5)),
                0.95: float(np.quantile(self_transition_probs, 0.95)),
            }
            persistence_thresholds = {
                "high": quantiles[0.95],
                "low": quantiles[0.25],
            }
        else:
            quantiles = {}
            persistence_thresholds = {"high": 0.99, "low": 0.6}

        statistics_cache = {
            "dirichlet_alpha": {"tas": tas_priors, "nas": nas_priors},
            "transition_regularizer": transition_regularizer,
            "persistence_thresholds": persistence_thresholds,
            "persistence_quantiles": quantiles,
            "dwell_times": dwell_times,
            "disagreement_rate": disagreement_rate,
        }

        return self._ensure_statistics_cache(statistics_cache)

    def _get_dirichlet_alpha(self, n_classes: int, annotator: str) -> Optional[np.ndarray]:
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

    def _map_using_gmm(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
        """Map labels using GMM with M1 optimizations and matrix operations."""
        with tprint_timer(f"GMM mapping with target_k={target_k}"):
            try:
                # Validate input data
                features = validate_finite(features, "gmm_features")
                tas_assignments = validate_finite(tas_assignments, "tas_assignments")
                nas_assignments = validate_finite(nas_assignments, "nas_assignments")
                
                # Use memory checkpoint for large datasets
                with memory_checkpoint("gmm_mapping"):
                    # Optimize GMM fitting for M1 if available
                    if M1_HARDWARE_AVAILABLE and self.cpu_optimizer:
                        gmm = self.cpu_optimizer.optimize_gmm_fitting(features, target_k)
                    else:
                        gmm = GaussianMixture(n_components=target_k, random_state=42)
                        gmm.fit(features)
                    
                    centroids = gmm.means_
                    
                    # Validate centroids
                    centroids = validate_finite(centroids, "gmm_centroids")

                    tas_mapping: Dict[int, int] = {}
                    nas_mapping: Dict[int, int] = {}

                    # Use optimized matrix operations for distance calculations
                    if MATRIX_OPERATIONS_AVAILABLE:
                        # Batch process label mappings
                        tas_labels = list(set(tas_assignments.tolist()))
                        nas_labels = list(set(nas_assignments.tolist()))
                        
                        # Process TAS labels
                        for label in tas_labels:
                            mapped_label = self._nearest_centroid_label_optimized(
                                label, tas_assignments, features, centroids, target_k
                            )
                            tas_mapping[label] = mapped_label

                        # Process NAS labels
                        for label in nas_labels:
                            mapped_label = self._nearest_centroid_label_optimized(
                                label, nas_assignments, features, centroids, target_k
                            )
                            nas_mapping[label] = mapped_label
                    else:
                        # Fallback to original method
                        for label in set(tas_assignments.tolist()):
                            mapped_label = self._nearest_centroid_label(label, tas_assignments, features, centroids, target_k)
                            tas_mapping[label] = mapped_label

                        for label in set(nas_assignments.tolist()):
                            mapped_label = self._nearest_centroid_label(label, nas_assignments, features, centroids, target_k)
                            nas_mapping[label] = mapped_label

                    # Create mapped arrays with validation
                    tas_mapped = np.array([tas_mapping.get(label, label % target_k) for label in tas_assignments])
                    nas_mapped = np.array([nas_mapping.get(label, label % target_k) for label in nas_assignments])
                    
                    # Validate results
                    tas_mapped = validate_finite(tas_mapped, "tas_mapped")
                    nas_mapped = validate_finite(nas_mapped, "nas_mapped")
                    
                    # Log mapping statistics
                    tprint_structured({
                        "target_k": target_k,
                        "tas_unique_mapped": len(set(tas_mapped)),
                        "nas_unique_mapped": len(set(nas_mapped)),
                        "tas_mapping_size": len(tas_mapping),
                        "nas_mapping_size": len(nas_mapping)
                    })

                    tprint_success(f"GMM mapping completed: {len(tas_mapping)} TAS, {len(nas_mapping)} NAS mappings")
                    return tas_mapped, nas_mapped, tas_mapping, nas_mapping
                    
            except Exception as exc:
                tprint_error(f"Failed to map using GMM: {exc}")
                raise

    def _nearest_centroid_label_optimized(
        self,
        label: int,
        assignments: np.ndarray,
        features: np.ndarray,
        centroids: np.ndarray,
        target_k: int,
    ) -> int:
        """Optimized nearest centroid calculation using matrix operations."""
        try:
            mask = assignments == label
            if not np.any(mask):
                return int(label % target_k)

            label_features = features[mask]
            
            # Use optimized matrix operations if available
            if MATRIX_OPERATIONS_AVAILABLE and self.vectorized_core:
                # Vectorized distance calculation
                distances = self.vectorized_core.calculate_distances_to_centroids(
                    label_features, centroids
                )
            else:
                # Fallback to standard calculation
                distances = np.linalg.norm(label_features[:, np.newaxis] - centroids, axis=2)
            
            # Use safe mean calculation
            mean_distances = safe_mean(distances, axis=0)
            return int(np.argmin(mean_distances))
            
        except Exception as exc:
            tprint_warning(f"Optimized centroid calculation failed for label {label}: {exc}")
            # Fallback to original method
            return self._nearest_centroid_label(label, assignments, features, centroids, target_k)

    def _nearest_centroid_label(
        self,
        label: int,
        assignments: np.ndarray,
        features: np.ndarray,
        centroids: np.ndarray,
        target_k: int,
    ) -> int:
        """Original nearest centroid calculation with safe operations."""
        try:
            mask = assignments == label
            if not np.any(mask):
                return int(label % target_k)

            label_features = features[mask]
            
            # Validate features
            label_features = validate_finite(label_features, "label_features")
            
            # Calculate distances with safe operations
            distances = np.linalg.norm(label_features[:, np.newaxis] - centroids, axis=2)
            distances = validate_finite(distances, "distances")
            
            # Use safe mean calculation
            mean_distances = safe_mean(distances, axis=0)
            return int(np.argmin(mean_distances))
            
        except Exception as exc:
            tprint_warning(f"Centroid calculation failed for label {label}: {exc}")
            return int(label % target_k)

    def _create_abstain_mapping(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], int]:
        tas_mapping: Dict[int, int] = {}
        nas_mapping: Dict[int, int] = {}
        abstain_value = target_k

        for label in set(tas_assignments.tolist()):
            tas_mapping[label] = label if 0 <= label < target_k else abstain_value

        for label in set(nas_assignments.tolist()):
            nas_mapping[label] = label if 0 <= label < target_k else abstain_value

        tas_mapped = np.array([tas_mapping.get(label, abstain_value) for label in tas_assignments])
        nas_mapped = np.array([nas_mapping.get(label, abstain_value) for label in nas_assignments])

        tprint(
            f"Abstain mapping applied – TAS unique: {len(set(tas_mapped))}, NAS unique: {len(set(nas_mapped))}",
            "INFO"
        )

        return tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value

    def _initialize_confusion_matrices(self, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
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
        """Enhanced E-step with matrix operations and M1 optimizations."""
        with tprint_timer("E-step calculation"):
            try:
                n_samples, n_classes = posteriors.shape
                abstain_value = mapping_info.get("abstain_value")

                # Validate input arrays
                tas_mapped = validate_numeric_array(tas_mapped, "tas_mapped")
                nas_mapped = validate_numeric_array(nas_mapped, "nas_mapped")
                tas_confusion = validate_numeric_array(tas_confusion, "tas_confusion")
                nas_confusion = validate_numeric_array(nas_confusion, "nas_confusion")
                class_priors = validate_numeric_array(class_priors, "class_priors")

                # Use memory checkpoint for large datasets
                with memory_checkpoint("e_step_calculation"):
                    # Optimize with matrix operations if available
                    if (
                        MATRIX_OPERATIONS_AVAILABLE
                        and self.matrix_ops is not None
                        and hasattr(self.matrix_ops, "vectorized_e_step")
                    ):
                        # Vectorized E-step calculation - update in place
                        posteriors[:] = self.matrix_ops.vectorized_e_step(
                            tas_mapped, nas_mapped, tas_confusion, nas_confusion,
                            class_priors, abstain_value
                        )
                    else:
                        # Original E-step with safe operations using math_validation
                        for i in range(n_samples):
                            tas_observation = tas_mapped[i]
                            nas_observation = nas_mapped[i]

                            for true_class in range(n_classes):
                                # Safe factor calculations with validation
                                tas_factor = validate_finite(
                                    tas_confusion[true_class, tas_observation]
                                    if tas_observation < n_classes
                                    else 1.0,
                                    "tas_factor"
                                )
                                nas_factor = validate_finite(
                                    nas_confusion[true_class, nas_observation]
                                    if nas_observation < n_classes
                                    else 1.0,
                                    "nas_factor"
                                )

                                if abstain_value is not None and (
                                    tas_observation == abstain_value or nas_observation == abstain_value
                                ):
                                    tas_factor = 1.0 if tas_observation == abstain_value else tas_factor
                                    nas_factor = 1.0 if nas_observation == abstain_value else nas_factor

                                # Use safe multiplication with validation
                                prior_value = validate_finite(class_priors[true_class], "class_prior")
                                
                                # Check for zero or negative values that would cause issues
                                if prior_value <= 0:
                                    tprint_warning(f"Invalid prior value {prior_value} for class {true_class}")
                                    prior_value = 1e-10  # Small positive value
                                
                                if tas_factor <= 0:
                                    tprint_warning(f"Invalid TAS factor {tas_factor} for class {true_class}")
                                    tas_factor = 1e-10
                                    
                                if nas_factor <= 0:
                                    tprint_warning(f"Invalid NAS factor {nas_factor} for class {true_class}")
                                    nas_factor = 1e-10
                                
                                # Calculate product with proper validation
                                product = prior_value * tas_factor * nas_factor
                                product = validate_finite(product, "posterior_product")
                                
                                # Ensure product is positive
                                if product <= 0:
                                    tprint_warning(f"Non-positive posterior product {product} for class {true_class}")
                                    product = 1e-10
                                
                                posteriors[i, true_class] = product

                    # Normalize posteriors with safe operations and validation - update in place
                    row_sums = posteriors.sum(axis=1, keepdims=True)
                    row_sums = validate_numeric_array(row_sums, "row_sums")

                    # Handle zero sums by setting uniform distribution
                    zero_sum_mask = (row_sums == 0.0).flatten()
                    if np.any(zero_sum_mask):
                        tprint_warning(f"Found {np.sum(zero_sum_mask)} samples with zero posterior sums")
                        posteriors[zero_sum_mask] = 1.0 / n_classes  # Uniform distribution
                        row_sums[zero_sum_mask] = 1.0

                    # Ensure all row sums are positive
                    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)

                    # Normalize with safe division using numpy operations
                    posteriors[:] = posteriors / row_sums
                    posteriors[:] = np.nan_to_num(posteriors, nan=1.0 / n_classes, posinf=1.0 / n_classes, neginf=1.0 / n_classes)

                    # Ensure posteriors are valid probabilities and sum to 1
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
        priors = posteriors.mean(axis=0)
        priors_sum = priors.sum()
        if priors_sum == 0:
            return np.ones_like(priors) / len(priors)
        return priors / priors_sum

    def _update_confusion_matrix(
        self, posteriors: np.ndarray, mapped_assignments: np.ndarray, n_classes: int
    ) -> np.ndarray:
        confusion = np.zeros((n_classes, n_classes))
        for true_class in range(n_classes):
            for observed_class in range(n_classes):
                mask = mapped_assignments == observed_class
                if np.any(mask):
                    confusion[true_class, observed_class] = posteriors[mask, true_class].sum()

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
        tas_change = np.abs(new_tas - old_tas).max()
        nas_change = np.abs(new_nas - old_nas).max()
        prior_change = np.abs(new_priors - old_priors).max()
        return max(tas_change, nas_change, prior_change) < tolerance


class RegimeOptimizationService:
    """Service responsible for regime optimization, scoring and smoothing."""

    def __init__(
        self,
        label_fusion_service: Optional[LabelFusionService],
        score_calculator: Callable[[np.ndarray, np.ndarray], float],
        logger: Callable[[str, str], None] = _default_logger,
    ) -> None:
        # Create LabelFusionService if None provided
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
        """Run Dawid–Skene fusion then score the resulting assignments."""

        tprint("Starting progressive regime optimization", "INFO")
        fusion_result = self._label_fusion_service.run_dawid_skene(
            tas_assignments,
            nas_assignments,
            optimal_k,
            features,
        )

        _ = market_data  # retained for signature compatibility

        mapped_assignments = self._map_to_optimal_k(fusion_result.assignments, features, optimal_k)

        initial_score = self._score_calculator(features, mapped_assignments)

        optimization_metrics = {
            "initial_score": initial_score,
            "final_score": initial_score,
            "improvement": 0.0,
            "iterations": 1,
            "optimal_k": optimal_k,
            "method": "data_driven_optimization",
            "fusion_metadata": fusion_result.metadata,
        }

        tprint_success(
            f"Progressive optimization completed – Score: {initial_score:.3f}"
        )

        return mapped_assignments, optimization_metrics, fusion_result.metadata

    def apply_hmm_smoothing(
        self, features: np.ndarray, assignments: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply HMM-based smoothing with simple fallback."""

        try:
            model = self._initialize_hmm(features, assignments)
            model.fit(features)
            smoothed_assignments = model.predict(features)
            metadata = self._build_smoothing_metadata(model, assignments, smoothed_assignments)
            tprint_success(
                f"HMM smoothing completed – {model.n_components} clusters"
            )
            return smoothed_assignments, metadata
        except Exception as exc:  # pragma: no cover - safety fallback
            tprint_warning(
                f"HMM smoothing failed ({exc}), using simple smoothing fallback"
            )
            smoothed = self._simple_temporal_smoothing(assignments)
            return smoothed, {"method": "simple_fallback", "error": str(exc)}

    def _map_to_optimal_k(
        self, assignments: np.ndarray, features: np.ndarray, optimal_k: int
    ) -> np.ndarray:
        if len(set(assignments.tolist())) == optimal_k:
            tprint_success(
                f"Assignments already match optimal K={optimal_k}"
            )
            return assignments

        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm.fit(features)
        mapped = gmm.predict(features)
        tprint_success(
            f"Assignments remapped via GMM – clusters: {len(set(mapped.tolist()))}"
        )
        return mapped

    def _initialize_hmm(self, features: np.ndarray, assignments: np.ndarray) -> "hmm.GaussianHMM":
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
        high_threshold = self._label_fusion_service.get_persistence_threshold("high", default=0.99)
        low_threshold = self._label_fusion_service.get_persistence_threshold("low", default=0.6)
        persistence_quantiles = self._label_fusion_service.get_persistence_quantiles()

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
            "persistence_quantiles": persistence_quantiles,
        }

        if low_persistence_regimes:
            tprint_warning(
                f"Low persistence regimes detected: {low_persistence_regimes}"
            )

        return metadata

    def _simple_temporal_smoothing(self, assignments: np.ndarray) -> np.ndarray:
        smoothed = assignments.copy()
        for idx in range(1, len(assignments) - 1):
            if assignments[idx] != assignments[idx - 1] and assignments[idx] != assignments[idx + 1]:
                window = assignments[max(0, idx - 2) : min(len(assignments), idx + 3)]
                values, counts = np.unique(window, return_counts=True)
                smoothed[idx] = int(values[np.argmax(counts)])
        return smoothed

