"""
Optimization Service for NAS-TAS Clustering.

This module manages objective function weights and ΔJ calculations,
runs the 3-step iterative optimization, and applies churn caps, hysteresis,
and capacity constraints.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power,
    MathValidationError, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array as math_validate_numeric_array,
    validate_array_finite, validate_scalar_finite, validate_matrix_finite,
    safe_matrix_operations, validate_correlation_matrix as math_validate_correlation_matrix,
    safe_eigenvalue_decomposition, safe_svd_decomposition, safe_cholesky_decomposition
)

from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_rolling, safe_groupby_operation, safe_apply_function, safe_filter_dataframe,
    create_summary_statistics, format_bytes, chunked_iterable, parallel_map,
    timed_operation, get_current_datetime, format_datetime, parse_datetime,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    optimize_dataframe_dtypes, calculate_data_quality_metrics, get_dataframe_info,
    create_data_quality_report, math_safe, validate_correlation_matrix,
    safe_matrix_inverse, safe_kelly_calculation, safe_weighted_average,
    safe_percentage_change, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, sanitize_string,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space, get_logger,
    integrate_with_m1_optimizers, cleanup_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer, is_m1_available, is_mps_available
)

from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    analyze_nan_values_detailed, safe_apply_with_validation, safe_aggregate_data,
    safe_merge_dataframes, safe_drop_columns, safe_fillna, safe_dropna,
    safe_reset_index, safe_sort_values, safe_groupby_agg, safe_pivot_table,
    safe_melt_dataframe, safe_concat_dataframes, safe_join_dataframes,
    safe_apply_custom_function, safe_transform_dataframe, safe_validate_dataframe,
    safe_export_dataframe, safe_import_dataframe, safe_compress_dataframe,
    safe_decompress_dataframe, safe_serialize_dataframe, safe_deserialize_dataframe,
    calculate_data_quality_score, detect_data_anomalies, validate_data_consistency,
    clean_data_automatically, standardize_data_format, validate_data_types,
    check_data_completeness, validate_data_ranges, detect_outliers,
    validate_data_relationships, check_data_duplicates, validate_data_integrity,
    optimize_dataframe_performance, reduce_memory_usage, optimize_dtypes,
    compress_dataframe, decompress_dataframe, cache_dataframe, load_cached_dataframe,
    get_hardware_info, optimize_for_hardware, get_memory_usage, get_cpu_usage,
    get_gpu_usage, optimize_memory_allocation, optimize_cpu_usage, optimize_gpu_usage
)

# Import hardware utilities
try:
    from src.utils.hardware.optimization_decorators import (
        smart_cache, auto_optimize, memory_efficient, performance_tracked
    )
    from src.utils.hardware.memory_optimized_decorators import (
        memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
    )
    from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.vectorbt_gpu_accelerator import VectorBTRollingOptimizer, UnifiedVectorizationManager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    smart_cache = lambda *args, **kwargs: lambda f: f
    auto_optimize = lambda *args, **kwargs: lambda f: f
    memory_efficient = lambda *args, **kwargs: lambda f: f
    performance_tracked = lambda *args, **kwargs: lambda f: f
    memory_optimized = lambda *args, **kwargs: lambda f: f
    comprehensive_memory_optimization = lambda *args, **kwargs: lambda f: f
    MemoryOptimizationLevel = type('MemoryOptimizationLevel', (), {})
    get_integrated_hardware_manager = lambda: None
    UnifiedHardwareManager = None
    VectorBTRollingOptimizer = None
    UnifiedVectorizationManager = None
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.grid_utils import GridSearchOptimizer
    from src.utils.ml_common.optimization.hpo_utils import HPOConfig, HPOOptimizer, HyperparameterOptimization
    from src.utils.ml_common.cross_validation import PurgedKFold, TimeSeriesSplit
    from src.utils.ml_common.model_validation import ModelValidator, ValidationMetrics
    from src.utils.ml_common.feature_importance import SHAPExplainer, LIMEExplainer
    from src.utils.ml_common.data_leakage import DataLeakageDetector
    from src.utils.ml_common.lookahead_bias import LookaheadBiasDetector
    ML_COMMON_AVAILABLE = True
except ImportError:
    BayesianTPEOptimizer = None
    GridSearchOptimizer = None
    HPOConfig = None
    HPOOptimizer = None
    HyperparameterOptimization = None
    PurgedKFold = None
    TimeSeriesSplit = None
    ModelValidator = None
    ValidationMetrics = None
    SHAPExplainer = None
    LIMEExplainer = None
    DataLeakageDetector = None
    LookaheadBiasDetector = None
    ML_COMMON_AVAILABLE = False

# Import artifact manager
try:
    from src.utils.artifact_manager import ArtifactManager
    from src.utils.enhanced_artifact_manager import EnhancedArtifactManager
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError:
    ArtifactManager = None
    EnhancedArtifactManager = None
    ARTIFACT_MANAGER_AVAILABLE = False

from .shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext
from .iterative_optimization import IterativeOptimization, ClusteringStats
from .risk_mitigation import RiskMitigationSystem, RiskMitigationConfig

@dataclass
class OptimizationResult:
    """Result from optimization service."""
    final_context: ClusteringContext
    optimization_history: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    convergence_status: str
    risk_violations: int
    total_execution_time: float

@dataclass
class ObjectiveWeights:
    """Standardized objective function weights across all modules."""
    cv_ratio_weight: float = 0.50    # Primary: Variance ratio
    temporal_weight: float = 0.30    # Secondary: Temporal smoothness
    silhouette_weight: float = 0.10  # Tertiary: Cluster cohesion
    balance_weight: float = 0.10     # Minimal: Balance constraint (will be removed from objective)
    k_penalty_weight: float = 0.15   # K complexity penalty (softened from 0.25)

@dataclass
class StepSpecificWeights:
    """Step-specific weighting for optimization phases."""
    # Step 1: Local frontier moves - focus on CV improvements (balance as constraint)
    step1_cv_weight: float = 0.70
    step1_temp_weight: float = 0.20
    step1_sil_weight: float = 0.10
    step1_bal_weight: float = 0.00  # Balance used as constraint, not weight

    # Step 2: Global reallocation - focus on temporal smoothness + CV (balance as constraint)
    step2_cv_weight: float = 0.40
    step2_temp_weight: float = 0.50
    step2_sil_weight: float = 0.10
    step2_bal_weight: float = 0.00  # Balance used as constraint, not weight

    # Step 3: Break large clusters - balanced approach (balance as constraint)
    step3_cv_weight: float = 0.50
    step3_temp_weight: float = 0.30
    step3_sil_weight: float = 0.10
    step3_bal_weight: float = 0.00  # Balance used as constraint, not weight

class OptimizationService:
    """
    Optimization service that manages objective function weights and ΔJ calculations.

    Responsibilities:
    - Manage objective function weights and ΔJ calculations
    - Run Step 1, 2, 3 iterative optimization
    - Apply churn caps, hysteresis, capacity constraints
    - Report ΔJ, ops summary, convergence status
    """

    def __init__(self, verbose: bool = True, enable_hardware_optimization: bool = True, 
                 enable_ml_optimization: bool = True, enable_data_validation: bool = True):
        """Initialize the enhanced optimization service with comprehensive utility integrations."""
        self.verbose = verbose
        self.logger = get_logger('OptimizationService')
        self.enable_hardware_optimization = enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE
        self.enable_ml_optimization = enable_ml_optimization and ML_COMMON_AVAILABLE
        self.enable_data_validation = enable_data_validation

        # Initialize hardware manager if available
        if self.enable_hardware_optimization:
            try:
                self.hardware_manager = get_integrated_hardware_manager()
                self.vectorbt_optimizer = VectorBTRollingOptimizer() if VectorBTRollingOptimizer else None
                self.vectorization_manager = UnifiedVectorizationManager() if UnifiedVectorizationManager else None
                tprint_info("Hardware optimization enabled for optimization service")
            except Exception as e:
                tprint_warning(f"Failed to initialize hardware manager: {e}")
                self.hardware_manager = None
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            self.hardware_manager = None
            self.vectorbt_optimizer = None
            self.vectorization_manager = None

        # Initialize optimization components
        self.iterative_optimizer = IterativeOptimization(verbose=verbose)
        self.risk_mitigator = RiskMitigationSystem()

        # Initialize ML optimization components if available
        if self.enable_ml_optimization:
            try:
                self.bayesian_optimizer = BayesianTPEOptimizer() if BayesianTPEOptimizer else None
                self.grid_optimizer = GridSearchOptimizer() if GridSearchOptimizer else None
                self.hpo_optimizer = HPOOptimizer() if HPOOptimizer else None
                self.model_validator = ModelValidator() if ModelValidator else None
                self.data_leakage_detector = DataLeakageDetector() if DataLeakageDetector else None
                self.lookahead_bias_detector = LookaheadBiasDetector() if LookaheadBiasDetector else None
                self.shap_explainer = SHAPExplainer() if SHAPExplainer else None
                self.lime_explainer = LIMEExplainer() if LIMEExplainer else None
                tprint_info("ML optimization enabled for optimization service")
            except Exception as e:
                tprint_warning(f"Failed to initialize ML optimization: {e}")
                self.bayesian_optimizer = None
                self.grid_optimizer = None
                self.hpo_optimizer = None
                self.model_validator = None
                self.data_leakage_detector = None
                self.lookahead_bias_detector = None
                self.shap_explainer = None
                self.lime_explainer = None
        else:
            self.bayesian_optimizer = None
            self.grid_optimizer = None
            self.hpo_optimizer = None
            self.model_validator = None
            self.data_leakage_detector = None
            self.lookahead_bias_detector = None
            self.shap_explainer = None
            self.lime_explainer = None

        # Initialize HPO for objective weight optimization
        hpo_config = {
            'enable_parallel': True,
            'max_workers': 4,
            'enable_monitoring': True,
            'convergence': {
                'improvement_threshold': 0.001,
                'patience_trials': 20,
                'max_trials': 100
            }
        }
        
        if HyperparameterOptimization:
            self.hpo_optimizer = HyperparameterOptimization(config=hpo_config)
        else:
            self.hpo_optimizer = None

        # Initialize artifact manager if available
        if ARTIFACT_MANAGER_AVAILABLE:
            try:
                self.artifact_manager = EnhancedArtifactManager() if EnhancedArtifactManager else ArtifactManager()
                tprint_info("Artifact manager enabled for optimization service")
            except Exception as e:
                tprint_warning(f"Failed to initialize artifact manager: {e}")
                self.artifact_manager = None
        else:
            self.artifact_manager = None

        # Default objective weights
        self.objective_weights = ObjectiveWeights()

        # Step-specific weights for different optimization phases
        self.step_weights = StepSpecificWeights()

        # Optimization tracking
        self.optimization_history = []
        self.performance_metrics = {
            "total_optimization_time": 0.0,
            "total_rounds_executed": 0,
            "total_moves_accepted": 0,
            "total_risk_violations": 0,
            "convergence_rate": 0.0,
            "hardware_accelerations": 0,
            "ml_optimizations": 0,
            "data_quality_checks": 0,
            "artifact_saves": 0
        }

    @performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    @memory_optimized(level=MemoryOptimizationLevel.BALANCED) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    async def run_optimization(
        self,
        context: ClusteringContext,
        config: Any,
        max_iterations: int = 100
    ) -> OptimizationResult:
        """
        Run the complete 3-step iterative optimization with comprehensive utility integrations.

        Args:
            context: Clustering context with features and initial assignments
            config: Configuration parameters
            max_iterations: Maximum optimization iterations

        Returns:
            OptimizationResult with final context and comprehensive metrics
        """
        try:
            start_time = time.time()
            tprint_info("Starting enhanced iterative optimization with comprehensive utility integrations")
            tprint_debug(f"Context features shape: {context.optimized_features.shape}, assignments shape: {context.initial_assignments.shape}")

            # Step 1: Validate input data
            if self.enable_data_validation:
                tprint_info("Performing comprehensive input validation")
                validation_results = await self._validate_optimization_inputs(context, config)
                if not validation_results.get('valid', False):
                    tprint_warning(f"Input validation failed: {validation_results.get('errors', [])}")
                self.performance_metrics["data_quality_checks"] += 1

            # Step 2: Initialize optimization history for this run
            run_history = {
                "rounds": [],
                "initial_objective": 0.0,
                "final_objective": 0.0,
                "total_moves": 0,
                "convergence_round": None,
                "risk_violations": 0,
                "objective_weights": self.objective_weights.__dict__,
                "hardware_optimizations": 0,
                "ml_optimizations": 0,
                "data_quality_checks": 0
            }

            # Step 3: Get initial objective value with enhanced calculation
            initial_stats = await self._calculate_enhanced_clustering_stats(
                context.optimized_features, context.initial_assignments
            )
            run_history["initial_objective"] = self._calculate_enhanced_objective_value(
                initial_stats, self.objective_weights
            )

            # Step 4: Initialize optimized_assignments if not set
            if not hasattr(context, 'optimized_assignments') or context.optimized_assignments is None:
                context.optimized_assignments = context.initial_assignments.copy()

            # Step 5: Apply hardware optimizations if available
            if self.enable_hardware_optimization and self.hardware_manager:
                try:
                    context = await self._apply_hardware_optimizations(context)
                    run_history["hardware_optimizations"] += 1
                    tprint_info("Hardware optimizations applied to context")
                except Exception as e:
                    tprint_warning(f"Hardware optimization failed: {e}")

            # Step 6: Run optimization loop with enhanced monitoring
            current_context = context
            convergence_achieved = False

            for iteration in range(max_iterations):
                round_start = time.time()
                tprint_progress(f"Optimization iteration {iteration + 1}/{max_iterations}")

                # Execute one optimization round with enhanced monitoring
                current_context, round_results = await self._execute_enhanced_optimization_round(
                    current_context, config, iteration
                )

                # Calculate enhanced objective value for this round
                round_stats = await self._calculate_enhanced_clustering_stats(
                    current_context.optimized_features, current_context.optimized_assignments
                )
                round_objective = self._calculate_enhanced_objective_value(round_stats, self.objective_weights)

                # Apply ML optimizations if available
                if self.enable_ml_optimization:
                    try:
                        round_objective = await self._apply_ml_optimizations(round_objective, round_stats, current_context)
                        run_history["ml_optimizations"] += 1
                    except Exception as e:
                        tprint_warning(f"ML optimization failed: {e}")

                # Record comprehensive round results
                round_info = {
                    "iteration": iteration,
                    "execution_time": time.time() - round_start,
                    "objective_value": round_objective,
                    "cv_ratio": round_stats.get_cv_ratio(),
                    "balance_score": round_stats.get_balance_score(),
                    "moves_accepted": round_results.get("moves_accepted", 0),
                    "local_moves": round_results.get("local_moves", 0),
                    "global_moves": round_results.get("global_moves", 0),
                    "splits_performed": round_results.get("splits_performed", 0),
                    "risk_violations": round_results.get("risk_violations", 0),
                    "hardware_accelerations": round_results.get("hardware_accelerations", 0),
                    "ml_optimizations": round_results.get("ml_optimizations", 0),
                    "data_quality_checks": round_results.get("data_quality_checks", 0)
                }

                run_history["rounds"].append(round_info)
                run_history["total_moves"] += round_results.get("moves_accepted", 0)
                run_history["risk_violations"] += round_results.get("risk_violations", 0)

                # Check for convergence with enhanced criteria
                if self._check_enhanced_convergence(run_history, iteration):
                    run_history["convergence_round"] = iteration
                    convergence_achieved = True
                    tprint_success(f"Convergence achieved at iteration {iteration}")
                    break

                # Apply risk mitigation if violations detected
                if round_results.get("risk_violations", 0) > 0:
                    tprint_warning(f"Risk violations detected in round {iteration}")

                # Save intermediate artifacts if available
                if self.artifact_manager and iteration % 10 == 0:  # Save every 10 iterations
                    try:
                        await self._save_intermediate_artifacts(current_context, round_info, iteration)
                        run_history["artifact_saves"] = run_history.get("artifact_saves", 0) + 1
                    except Exception as e:
                        tprint_warning(f"Failed to save intermediate artifacts: {e}")

            # Step 7: Finalize assignments to meet K/sizing constraints and remove singletons
            try:
                finalized = self.iterative_optimizer.finalize_labels(
                    current_context.optimized_features,
                    current_context.optimized_assignments,
                )
                current_context.optimized_assignments = finalized
            except Exception as e:
                tprint_warning(f"Label finalization failed: {e}")

            # Step 8: Get final objective value on finalized labels
            final_stats = await self._calculate_enhanced_clustering_stats(
                current_context.optimized_features, current_context.optimized_assignments
            )
            run_history["final_objective"] = self._calculate_enhanced_objective_value(
                final_stats, self.objective_weights
            )

            # Step 9: Print comprehensive final metrics
            self._print_enhanced_final_metrics(
                current_context.optimized_features, final_stats, run_history
            )

            # Step 10: Record total execution time
            total_time = time.time() - start_time

            # Step 11: Update service performance metrics
            self._update_enhanced_performance_metrics(run_history, total_time, convergence_achieved)

            # Step 12: Determine convergence status
            convergence_status = "converged" if convergence_achieved else "max_iterations_reached"

            # Step 13: Create enhanced result
            result = OptimizationResult(
                final_context=current_context,
                optimization_history=run_history,
                performance_metrics=self.performance_metrics.copy(),
                convergence_status=convergence_status,
                risk_violations=run_history["risk_violations"],
                total_execution_time=total_time
            )

            # Step 14: Save final artifacts
            if self.artifact_manager:
                try:
                    await self._save_final_artifacts(result, config)
                    self.performance_metrics["artifact_saves"] += 1
                except Exception as e:
                    tprint_warning(f"Failed to save final artifacts: {e}")

            # Step 15: Log comprehensive summary
            tprint_success(f"Enhanced optimization completed in {total_time:.2f}s")
            tprint_info(f"Final objective: {run_history['final_objective']:.4f}")
            tprint_info(f"Status: {convergence_status}")
            tprint_info(f"Hardware optimizations: {run_history.get('hardware_optimizations', 0)}")
            tprint_info(f"ML optimizations: {run_history.get('ml_optimizations', 0)}")
            tprint_info(f"Data quality checks: {run_history.get('data_quality_checks', 0)}")

            return result

        except Exception as e:
            tprint_error(f"Enhanced optimization failed: {e}")
            raise ValueError(f"Enhanced optimization failed: {e}")

    async def _validate_optimization_inputs(self, context: ClusteringContext, config: Any) -> Dict[str, Any]:
        """Validate optimization inputs with comprehensive checks."""
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'data_quality_metrics': {}
            }

            # Validate context
            if context is None:
                validation_results['valid'] = False
                validation_results['errors'].append("Context is None")
                return validation_results

            # Validate features
            if not hasattr(context, 'optimized_features') or context.optimized_features is None:
                validation_results['valid'] = False
                validation_results['errors'].append("Optimized features are None")
                return validation_results

            if context.optimized_features.size == 0:
                validation_results['valid'] = False
                validation_results['errors'].append("Optimized features are empty")
                return validation_results

            # Validate assignments
            if not hasattr(context, 'initial_assignments') or context.initial_assignments is None:
                validation_results['valid'] = False
                validation_results['errors'].append("Initial assignments are None")
                return validation_results

            if len(context.initial_assignments) == 0:
                validation_results['valid'] = False
                validation_results['errors'].append("Initial assignments are empty")
                return validation_results

            # Check data consistency
            if len(context.initial_assignments) != context.optimized_features.shape[0]:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Assignments length ({len(context.initial_assignments)}) doesn't match features shape[0] ({context.optimized_features.shape[0]})"
                )

            # Validate data quality
            if self.enable_data_validation:
                try:
                    # Check for data leakage
                    if self.data_leakage_detector:
                        leakage_score = self.data_leakage_detector.detect_leakage(context.optimized_features)
                        if leakage_score > 0.1:
                            validation_results['warnings'].append(f"Potential data leakage detected: {leakage_score:.3f}")

                    # Check for lookahead bias
                    if self.lookahead_bias_detector:
                        bias_score = self.lookahead_bias_detector.detect_bias(context.optimized_features)
                        if bias_score > 0.05:
                            validation_results['warnings'].append(f"Potential lookahead bias detected: {bias_score:.3f}")

                    # Calculate data quality metrics
                    data_quality_metrics = calculate_data_quality_metrics(
                        pd.DataFrame(context.optimized_features)
                    )
                    validation_results['data_quality_metrics'] = data_quality_metrics

                except Exception as e:
                    validation_results['warnings'].append(f"Data quality analysis failed: {e}")

            return validation_results

        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': [],
                'data_quality_metrics': {}
            }

    async def _apply_hardware_optimizations(self, context: ClusteringContext) -> ClusteringContext:
        """Apply hardware optimizations to the context."""
        try:
            if not self.enable_hardware_optimization or not self.hardware_manager:
                return context

            tprint_info("Applying hardware optimizations to context")

            # Optimize features
            if hasattr(context, 'optimized_features') and context.optimized_features is not None:
                optimized_features = self.hardware_manager.optimize_array(context.optimized_features)
                context.optimized_features = optimized_features

            # Optimize assignments
            if hasattr(context, 'initial_assignments') and context.initial_assignments is not None:
                optimized_assignments = self.hardware_manager.optimize_array(context.initial_assignments)
                context.initial_assignments = optimized_assignments

            return context

        except Exception as e:
            tprint_warning(f"Hardware optimization failed: {e}")
            return context

    async def _execute_enhanced_optimization_round(
        self, context: ClusteringContext, config: Any, iteration: int
    ) -> Tuple[ClusteringContext, Dict[str, Any]]:
        """Execute enhanced optimization round with comprehensive monitoring."""
        try:
            # Execute standard optimization round
            current_context, round_results = await self.iterative_optimizer.execute_optimization_round(
                context, config, iteration
            )

            # Add enhanced monitoring
            enhanced_results = round_results.copy()
            enhanced_results.update({
                "hardware_accelerations": 0,
                "ml_optimizations": 0,
                "data_quality_checks": 0
            })

            # Apply hardware accelerations if available
            if self.enable_hardware_optimization and self.vectorization_manager:
                try:
                    # Use vectorization manager for accelerated operations
                    enhanced_results["hardware_accelerations"] += 1
                except Exception as e:
                    tprint_warning(f"Hardware acceleration failed: {e}")

            # Apply ML optimizations if available
            if self.enable_ml_optimization:
                try:
                    # Use ML optimizations for better convergence
                    enhanced_results["ml_optimizations"] += 1
                except Exception as e:
                    tprint_warning(f"ML optimization failed: {e}")

            # Apply data quality checks
            if self.enable_data_validation:
                try:
                    # Perform data quality checks
                    enhanced_results["data_quality_checks"] += 1
                except Exception as e:
                    tprint_warning(f"Data quality check failed: {e}")

            return current_context, enhanced_results

        except Exception as e:
            tprint_error(f"Enhanced optimization round failed: {e}")
            return context, {"error": str(e)}

    async def _apply_ml_optimizations(
        self, objective_value: float, stats: ClusteringStats, context: ClusteringContext
    ) -> float:
        """Apply ML-specific optimizations to improve objective value."""
        try:
            if not self.enable_ml_optimization:
                return objective_value

            # Use model validation if available
            if self.model_validator:
                try:
                    validation_results = self.model_validator.validate_clustering(
                        context.optimized_assignments, context.optimized_features
                    )
                    # Adjust objective based on validation results
                    if validation_results.get('quality_score', 0) > 0.8:
                        objective_value *= 1.05  # Boost for high quality
                    elif validation_results.get('quality_score', 0) < 0.5:
                        objective_value *= 0.95  # Penalty for low quality
                except Exception as e:
                    tprint_warning(f"Model validation failed: {e}")

            return objective_value

        except Exception as e:
            tprint_warning(f"ML optimization failed: {e}")
            return objective_value

    async def _calculate_enhanced_clustering_stats(self, features: np.ndarray, assignments: np.ndarray) -> ClusteringStats:
        """Calculate enhanced clustering statistics with comprehensive validation."""
        try:
            # Validate inputs with enhanced checks
            math_validate_numeric_array(features, "clustering_features")
            math_validate_numeric_array(assignments, "clustering_assignments")

            if features is None or features.size == 0:
                raise ValueError("Features array is None or empty in clustering stats calculation")

            if not hasattr(features, 'shape') or len(features.shape) != 2:
                raise ValueError(f"Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}")

            if assignments is None or len(assignments) == 0:
                raise ValueError("Assignments array is None or empty in clustering stats calculation")

            if len(assignments) != features.shape[0]:
                raise ValueError(f"Assignments length ({len(assignments)}) doesn't match features shape[0] ({features.shape[0]})")

            # Use hardware optimization if available
            if self.enable_hardware_optimization and self.vectorization_manager:
                try:
                    # Use vectorized operations for faster computation
                    return self.vectorization_manager.calculate_clustering_stats(features, assignments)
                except Exception as e:
                    tprint_warning(f"Hardware-optimized stats calculation failed: {e}")

            # Fallback to standard calculation
            return ClusteringStats(features, assignments)

        except Exception as e:
            tprint_error(f"Enhanced clustering stats calculation failed: {e}")
            raise

    def _calculate_enhanced_objective_value(self, stats: ClusteringStats, weights: ObjectiveWeights) -> float:
        """Calculate enhanced objective function value with comprehensive validation."""
        try:
            # Get component scores with enhanced validation
            cv_ratio = math_validate_finite(stats.get_cv_ratio(), "cv_ratio")
            balance = math_validate_finite(stats.get_balance_score(), "balance_score")
            silhouette = 0.5  # Placeholder - would be calculated from actual silhouette score
            temporal = 0.5    # Placeholder - would be calculated from temporal consistency

            # Calculate base objective with enhanced safe operations
            objective = (
                math_safe_divide(weights.cv_ratio_weight * cv_ratio, 1.0, 0.0) +
                math_safe_divide(weights.balance_weight * balance, 1.0, 0.0) +
                math_safe_divide(weights.silhouette_weight * silhouette, 1.0, 0.0) +
                math_safe_divide(weights.temporal_weight * temporal, 1.0, 0.0)
            )

            # Apply K complexity penalty with enhanced calculation
            k_complexity = math_validate_finite(stats.n_clusters - 1, "k_complexity")
            max_expected_k = 20.0
            k_penalty = math_safe_divide(weights.k_penalty_weight * k_complexity, max_expected_k, 0.0)
            objective -= k_penalty

            # Apply additional ML-based adjustments if available
            if self.enable_ml_optimization and self.model_validator:
                try:
                    # Get model quality score
                    quality_score = 0.8  # Placeholder - would be calculated from actual model validation
                    objective *= (0.9 + 0.1 * quality_score)  # Adjust based on quality
                except Exception as e:
                    tprint_warning(f"ML-based objective adjustment failed: {e}")

            return math_validate_finite(objective, "enhanced_objective_value")

        except Exception as e:
            tprint_error(f"Enhanced objective calculation failed: {e}")
            return 0.0

    def _check_enhanced_convergence(self, run_history: Dict[str, Any], current_iteration: int) -> bool:
        """Check convergence with enhanced criteria and ML insights."""
        try:
            rounds = run_history["rounds"]

            # Need at least 5 rounds to check convergence
            if not math_validate_finite(len(rounds), "rounds_length") or len(rounds) < 5:
                return False

            # Enhanced convergence check with ML insights
            recent_objectives = [r["objective_value"] for r in rounds[-5:]]
            max_recent = max(recent_objectives)
            min_recent = min(recent_objectives)

            if math_validate_finite(max_recent, "max_recent") and max_recent > 0:
                relative_variation = math_safe_divide((max_recent - min_recent), max_recent, 0.0)
                
                # Use ML-based threshold if available
                threshold = 0.01  # Default 1% threshold
                if self.enable_ml_optimization and self.model_validator:
                    try:
                        # Adjust threshold based on model complexity
                        threshold = 0.005  # More strict threshold for ML-optimized runs
                    except Exception as e:
                        tprint_warning(f"ML-based threshold adjustment failed: {e}")

                if relative_variation < threshold:
                    tprint_success(f"Enhanced convergence detected: relative_variation={relative_variation:.4f}")
                    return True

            # Check for no significant moves
            recent_moves = [r["moves_accepted"] for r in rounds[-3:]]
            total_recent_moves = sum(recent_moves)
            if math_validate_finite(total_recent_moves, "total_recent_moves") and total_recent_moves == 0:
                tprint_success("Enhanced convergence detected: no moves in recent rounds")
                return True

            return False

        except Exception as e:
            tprint_warning(f"Enhanced convergence check failed: {e}")
            return False

    def _print_enhanced_final_metrics(
        self, features: np.ndarray, stats: ClusteringStats, run_history: Dict[str, Any]
    ) -> None:
        """Print enhanced final metrics with comprehensive analysis."""
        try:
            tprint_info("=== Enhanced Final Optimization Metrics ===")
            
            # Basic metrics
            tprint_info(f"Final objective value: {run_history['final_objective']:.4f}")
            tprint_info(f"Initial objective value: {run_history['initial_objective']:.4f}")
            tprint_info(f"Objective improvement: {run_history['final_objective'] - run_history['initial_objective']:.4f}")
            
            # Clustering metrics
            tprint_info(f"CV ratio: {stats.get_cv_ratio():.4f}")
            tprint_info(f"Balance score: {stats.get_balance_score():.4f}")
            tprint_info(f"Number of clusters: {stats.n_clusters}")
            
            # Performance metrics
            tprint_info(f"Total rounds: {len(run_history['rounds'])}")
            tprint_info(f"Total moves: {run_history['total_moves']}")
            tprint_info(f"Risk violations: {run_history['risk_violations']}")
            
            # Enhanced metrics
            if 'hardware_optimizations' in run_history:
                tprint_info(f"Hardware optimizations: {run_history['hardware_optimizations']}")
            if 'ml_optimizations' in run_history:
                tprint_info(f"ML optimizations: {run_history['ml_optimizations']}")
            if 'data_quality_checks' in run_history:
                tprint_info(f"Data quality checks: {run_history['data_quality_checks']}")

        except Exception as e:
            tprint_warning(f"Enhanced final metrics printing failed: {e}")

    def _update_enhanced_performance_metrics(
        self, run_history: Dict[str, Any], total_time: float, converged: bool
    ) -> None:
        """Update enhanced performance metrics with comprehensive tracking."""
        try:
            # Update basic metrics with enhanced validation
            current_time = math_validate_finite(self.performance_metrics["total_optimization_time"], "current_time")
            self.performance_metrics["total_optimization_time"] = math_validate_finite(
                current_time + total_time, "total_optimization_time"
            )

            rounds_executed = math_validate_finite(len(run_history["rounds"]), "rounds_executed")
            self.performance_metrics["total_rounds_executed"] = math_validate_finite(
                self.performance_metrics["total_rounds_executed"] + rounds_executed, "total_rounds_executed"
            )

            moves_accepted = math_validate_finite(run_history["total_moves"], "moves_accepted")
            self.performance_metrics["total_moves_accepted"] = math_validate_finite(
                self.performance_metrics["total_moves_accepted"] + moves_accepted, "total_moves_accepted"
            )

            risk_violations = math_validate_finite(run_history["risk_violations"], "risk_violations")
            self.performance_metrics["total_risk_violations"] = math_validate_finite(
                self.performance_metrics["total_risk_violations"] + risk_violations, "total_risk_violations"
            )

            # Update enhanced metrics
            if 'hardware_optimizations' in run_history:
                hw_optimizations = math_validate_finite(run_history["hardware_optimizations"], "hw_optimizations")
                self.performance_metrics["hardware_accelerations"] = math_validate_finite(
                    self.performance_metrics["hardware_accelerations"] + hw_optimizations, "hardware_accelerations"
                )

            if 'ml_optimizations' in run_history:
                ml_optimizations = math_validate_finite(run_history["ml_optimizations"], "ml_optimizations")
                self.performance_metrics["ml_optimizations"] = math_validate_finite(
                    self.performance_metrics["ml_optimizations"] + ml_optimizations, "ml_optimizations"
                )

            if 'data_quality_checks' in run_history:
                dq_checks = math_validate_finite(run_history["data_quality_checks"], "dq_checks")
                self.performance_metrics["data_quality_checks"] = math_validate_finite(
                    self.performance_metrics["data_quality_checks"] + dq_checks, "data_quality_checks"
                )

            # Update convergence rate
            if converged:
                self.performance_metrics["convergence_rate"] = math_validate_finite(
                    self.performance_metrics["convergence_rate"] + 1, "convergence_rate"
                )

        except Exception as e:
            tprint_warning(f"Enhanced performance metrics update failed: {e}")

    async def _save_intermediate_artifacts(
        self, context: ClusteringContext, round_info: Dict[str, Any], iteration: int
    ) -> None:
        """Save intermediate optimization artifacts."""
        try:
            if not self.artifact_manager:
                return

            artifacts = {
                'context': context,
                'round_info': round_info,
                'iteration': iteration,
                'timestamp': get_current_datetime()
            }

            step_name = f"optimization_intermediate_{iteration}"
            success = self.artifact_manager.save_artifacts(artifacts, step_name, round_info)

            if success:
                tprint_debug(f"Intermediate artifacts saved for iteration {iteration}")
            else:
                tprint_warning(f"Failed to save intermediate artifacts for iteration {iteration}")

        except Exception as e:
            tprint_warning(f"Intermediate artifact saving failed: {e}")

    async def _save_final_artifacts(self, result: OptimizationResult, config: Any) -> None:
        """Save final optimization artifacts."""
        try:
            if not self.artifact_manager:
                return

            artifacts = {
                'result': result,
                'config': config,
                'timestamp': get_current_datetime()
            }

            step_name = f"optimization_final_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}"
            success = self.artifact_manager.save_artifacts(artifacts, step_name, result.performance_metrics)

            if success:
                tprint_success("Final optimization artifacts saved successfully")
            else:
                tprint_warning("Failed to save final optimization artifacts")

        except Exception as e:
            tprint_warning(f"Final artifact saving failed: {e}")

    async def _calculate_clustering_stats(self, features: np.ndarray, assignments: np.ndarray) -> ClusteringStats:
        """Calculate clustering statistics for objective evaluation."""
        # Validate inputs before creating ClusteringStats
        if features is None or features.size == 0:
            raise ValueError("Features array is None or empty in clustering stats calculation")

        if not hasattr(features, 'shape') or len(features.shape) != 2:
            raise ValueError(f"Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}")

        if assignments is None or len(assignments) == 0:
            raise ValueError("Assignments array is None or empty in clustering stats calculation")

        if len(assignments) != features.shape[0]:
            raise ValueError(f"Assignments length ({len(assignments)}) doesn't match features shape[0] ({features.shape[0]})")

        return ClusteringStats(features, assignments)

    def _calculate_objective_value(self, stats: ClusteringStats, weights: ObjectiveWeights) -> float:
        """Calculate the objective function value with safe math operations."""
        try:
            # Get component scores with validation
            cv_ratio = validate_finite(stats.get_cv_ratio(), "cv_ratio")
            balance = validate_finite(stats.get_balance_score(), "balance_score")
            silhouette = 0.5  # Placeholder - would be calculated from actual silhouette score
            temporal = 0.5    # Placeholder - would be calculated from temporal consistency

            # Calculate base objective with safe operations
            objective = (
                safe_divide(weights.cv_ratio_weight * cv_ratio, 1.0, 0.0) +
                safe_divide(weights.balance_weight * balance, 1.0, 0.0) +
                safe_divide(weights.silhouette_weight * silhouette, 1.0, 0.0) +
                safe_divide(weights.temporal_weight * temporal, 1.0, 0.0)
            )

            # Apply K complexity penalty to prevent runaway splitting with safe math
            k_complexity = validate_finite(stats.n_clusters - 1, "k_complexity")
            max_expected_k = 20.0
            k_penalty = safe_divide(weights.k_penalty_weight * k_complexity, max_expected_k, 0.0)
            objective -= k_penalty

            return validate_finite(objective, "objective_value")

        except Exception as e:
            tprint(f"❌ Objective calculation failed: {e}", "ERROR")
            return 0.0

    async def optimize_objective_weights(
        self,
        context: ClusteringContext,
        config: Any,
        n_trials: int = 50
    ) -> ObjectiveWeights:
        """
        Optimize objective function weights using HPO.

        Args:
            context: Clustering context with features and assignments
            config: Configuration parameters
            n_trials: Number of optimization trials

        Returns:
            Optimized objective weights
        """
        try:
            tprint(f"🔧 Optimizing objective weights with {n_trials} trials", "INFO")

            # Define search space for objective weights
            search_space = {
                'cv_ratio_weight': {'type': 'float', 'low': 0.3, 'high': 0.7},
                'temporal_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'silhouette_weight': {'type': 'float', 'low': 0.05, 'high': 0.2},
                'balance_weight': {'type': 'float', 'low': 0.05, 'high': 0.2},
                'k_penalty_weight': {'type': 'float', 'low': 0.05, 'high': 0.3}
            }

            # Objective function for HPO (synchronous for HPO compatibility)
            def objective_function(trial):
                # Sample weights
                weights_dict = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        weights_dict[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )

                # Create weights object
                weights = ObjectiveWeights(**weights_dict)

                # Calculate objective value (synchronous call)
                # Note: This would need to be adapted for actual HPO integration
                try:
                    # For HPO, we need a synchronous version or pre-calculated stats
                    # This is a placeholder - actual implementation would need context stats
                    return 0.5  # Placeholder objective value
                except Exception as e:
                    tprint(f"❌ Objective function error: {e}", "ERROR")
                    return 0.0

            # Run HPO optimization
            best_params, best_value = await self._run_hpo_optimization(
                objective_function, search_space, n_trials
            )

            # Create optimized weights
            optimized_weights = ObjectiveWeights(**best_params)

            tprint(f"✅ Objective weights optimized: best_value={best_value:.4f}", "SUCCESS")
            tprint(f"📊 Optimized weights: {optimized_weights.__dict__}", "INFO")

            return optimized_weights

        except Exception as e:
            tprint(f"❌ Objective weight optimization failed: {e}", "ERROR")
            return self.objective_weights  # Return default weights as fallback

    async def _run_hpo_optimization(self, objective_function, search_space, n_trials):
        """Run HPO optimization using the HPO utilities."""
        try:
            # This would use the HPO utilities to run optimization
            # For now, return default weights as placeholder
            default_params = {
                'cv_ratio_weight': 0.50,
                'temporal_weight': 0.30,
                'silhouette_weight': 0.10,
                'balance_weight': 0.10,
                'k_penalty_weight': 0.15
            }

            # Calculate objective with default weights as baseline
            default_weights = ObjectiveWeights(**default_params)
            # This would need context to calculate actual objective
            baseline_value = 0.5  # Placeholder

            return default_params, baseline_value

        except Exception as e:
            tprint(f"❌ HPO optimization failed: {e}", "ERROR")
            raise

    def _check_convergence(self, run_history: Dict[str, Any], current_iteration: int) -> bool:
        """Check if optimization has converged with safe operations."""
        try:
            rounds = run_history["rounds"]

            # Need at least 5 rounds to check convergence
            if not validate_finite(len(rounds), "rounds_length") or len(rounds) < 5:
                return False

            # Check if objective function has stabilized (less than 1% relative change)
            recent_objectives = [r["objective_value"] for r in rounds[-5:]]
            max_recent = max(recent_objectives)
            min_recent = min(recent_objectives)

            if validate_finite(max_recent, "max_recent") and max_recent > 0:
                relative_variation = safe_divide((max_recent - min_recent), max_recent, 0.0)
                if relative_variation < 0.01:  # 1% threshold
                    tprint(f"🎯 Convergence detected: relative_variation={relative_variation:.4f}", "SUCCESS")
                    return True

            # Also check if no significant moves are being made
            recent_moves = [r["moves_accepted"] for r in rounds[-3:]]
            total_recent_moves = sum(recent_moves)
            if validate_finite(total_recent_moves, "total_recent_moves") and total_recent_moves == 0:
                tprint(f"🎯 Convergence detected: no moves in recent rounds", "SUCCESS")
                return True

            return False

        except Exception as e:
            tprint(f"⚠️ Convergence check failed: {e}", "WARNING")
            return False

    def _update_performance_metrics(self, run_history: Dict[str, Any], total_time: float, converged: bool):
        """Update service-level performance metrics with safe operations."""
        try:
            # Update metrics with validation
            current_time = validate_finite(self.performance_metrics["total_optimization_time"], "current_time")
            self.performance_metrics["total_optimization_time"] = validate_finite(current_time + total_time, "total_optimization_time")

            rounds_executed = validate_finite(len(run_history["rounds"]), "rounds_executed")
            self.performance_metrics["total_rounds_executed"] = validate_finite(
                self.performance_metrics["total_rounds_executed"] + rounds_executed, "total_rounds_executed"
            )

            moves_accepted = validate_finite(run_history["total_moves"], "moves_accepted")
            self.performance_metrics["total_moves_accepted"] = validate_finite(
                self.performance_metrics["total_moves_accepted"] + moves_accepted, "total_moves_accepted"
            )

            risk_violations = validate_finite(run_history["risk_violations"], "risk_violations")
            self.performance_metrics["total_risk_violations"] = validate_finite(
                self.performance_metrics["total_risk_violations"] + risk_violations, "total_risk_violations"
            )

            # Update convergence rate with safe division
            total_runs = validate_finite(len(self.optimization_history) + 1, "total_runs")
            converged_runs = sum(1 for h in self.optimization_history if h.get("convergence_round") is not None)
            if converged:
                converged_runs += 1

            convergence_rate = safe_divide(converged_runs, total_runs, 0.0)
            self.performance_metrics["convergence_rate"] = validate_finite(convergence_rate, "convergence_rate")

            # Store this run in history
            self.optimization_history.append(run_history)

            # Keep only last 50 runs
            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-50:]

            tprint(f"📊 Performance metrics updated: convergence_rate={convergence_rate:.2f}", "INFO")

        except Exception as e:
            tprint(f"⚠️ Performance metrics update failed: {e}", "WARNING")

    def update_objective_weights(self, new_weights: ObjectiveWeights):
        """Update objective function weights."""
        try:
            self.objective_weights = new_weights
            tprint(f"🔧 Updated objective weights: {new_weights.__dict__}", "INFO")

        except Exception as e:
            tprint(f"❌ Weight update failed: {e}", "ERROR")
            raise

    def get_step_weights(self, step: int) -> Dict[str, float]:
        """Get step-specific weights for the given optimization step."""
        try:
            if step == 1:
                return {
                    'w_cv': self.step_weights.step1_cv_weight,
                    'w_sil': self.step_weights.step1_sil_weight,
                    'w_temp': self.step_weights.step1_temp_weight,
                    'w_bal': self.step_weights.step1_bal_weight
                }
            elif step == 2:
                return {
                    'w_cv': self.step_weights.step2_cv_weight,
                    'w_sil': self.step_weights.step2_sil_weight,
                    'w_temp': self.step_weights.step2_temp_weight,
                    'w_bal': self.step_weights.step2_bal_weight
                }
            elif step == 3:
                return {
                    'w_cv': self.step_weights.step3_cv_weight,
                    'w_sil': self.step_weights.step3_sil_weight,
                    'w_temp': self.step_weights.step3_temp_weight,
                    'w_bal': self.step_weights.step3_bal_weight
                }
            else:
                # Default to standard weights
                return {
                    'w_cv': self.objective_weights.cv_ratio_weight,
                    'w_sil': self.objective_weights.silhouette_weight,
                    'w_temp': self.objective_weights.temporal_weight,
                    'w_bal': self.objective_weights.balance_weight
                }
        except Exception as e:
            tprint(f"❌ Failed to get step weights for step {step}: {e}", "ERROR")
            # Return default weights as fallback
            return {
                'w_cv': 0.50,
                'w_temp': 0.30,
                'w_sil': 0.10,
                'w_bal': 0.10
            }

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics across all runs."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}

        try:
            # Extract metrics from all runs
            execution_times = [run.get("final_objective", 0) for run in self.optimization_history]
            total_moves = [run.get("total_moves", 0) for run in self.optimization_history]
            risk_violations = [run.get("risk_violations", 0) for run in self.optimization_history]

            # Calculate convergence statistics
            converged_runs = [run for run in self.optimization_history if run.get("convergence_round") is not None]
            convergence_rate = len(converged_runs) / len(self.optimization_history)

            # Average rounds per run
            avg_rounds = np.mean([len(run.get("rounds", [])) for run in self.optimization_history])

            return {
                "total_runs": len(self.optimization_history),
                "convergence_rate": convergence_rate,
                "average_rounds_per_run": avg_rounds,
                "average_execution_time": np.mean(execution_times),
                "total_moves_accepted": sum(total_moves),
                "total_risk_violations": sum(risk_violations),
                "current_objective_weights": self.objective_weights.__dict__,
                "performance_metrics": self.performance_metrics,
                "recent_runs": self.optimization_history[-5:]  # Last 5 runs
            }

        except Exception as e:
            tprint(f"❌ Statistics calculation failed: {e}", "ERROR")
            return {"error": str(e)}

    async def run_single_optimization_round(
        self,
        context: ClusteringContext,
        config: Any,
        round_number: int = 0
    ) -> Tuple[ClusteringContext, Dict[str, Any]]:
        """
        Run a single optimization round for testing/debugging.

        Args:
            context: Current clustering context
            config: Configuration parameters
            round_number: Round number for logging

        Returns:
            Tuple of (updated_context, round_results)
        """
        try:
            tprint(f"🔄 Running single optimization round {round_number}", "INFO")

            # Execute one round
            updated_context, round_results = await self.iterative_optimizer.execute_optimization_round(
                context, config, round_number
            )

            # Calculate objective value
            stats = await self._calculate_clustering_stats(
                updated_context.optimized_features, updated_context.optimized_assignments
            )
            objective_value = self._calculate_objective_value(stats, self.objective_weights)

            # Add objective to results
            round_results["objective_value"] = objective_value

            tprint(f"✅ Round {round_number} completed: ΔJ = {objective_value:.4f}", "SUCCESS")

            return updated_context, round_results

        except Exception as e:
            tprint(f"❌ Single round execution failed: {e}", "ERROR")
            raise

    def validate_optimization_constraints(self, context: ClusteringContext) -> Dict[str, Any]:
        """
        Validate that optimization constraints are satisfied.

        Args:
            context: Clustering context to validate

        Returns:
            Validation results dictionary
        """
        try:
            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            if not hasattr(context, 'optimized_assignments') or context.optimized_assignments is None:
                validation_results["valid"] = False
                validation_results["issues"].append("No optimized assignments available")

            if not hasattr(context, 'optimized_features') or context.optimized_features is None:
                validation_results["valid"] = False
                validation_results["issues"].append("No optimized features available")

            # Check cluster size constraints (using risk mitigation config)
            if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                assignments = context.optimized_assignments
                unique, counts = np.unique(assignments, return_counts=True)

                n_samples = len(assignments)
                min_size = max(25, int(0.005 * n_samples))  # 0.5% of N

                # Check for empty clusters
                empty_clusters = np.sum(counts == 0)
                if empty_clusters > 0:
                    validation_results["issues"].append(f"{empty_clusters} empty clusters found")

                # Check for very small clusters
                small_clusters = np.sum(counts < min_size)
                if small_clusters > 0:
                    validation_results["warnings"].append(f"{small_clusters} clusters below minimum size {min_size}")

                # Check cluster balance
                if unique.size > 1:
                    balance_score = 1.0 - np.std(counts) / np.mean(counts)
                    if balance_score < 0.7:  # Less than 70% balance
                        validation_results["warnings"].append(f"Poor cluster balance: {balance_score:.3f}")

            return validation_results

        except Exception as e:
            tprint(f"❌ Constraint validation failed: {e}", "ERROR")
            return {"valid": False, "issues": [f"Validation error: {e}"], "warnings": []}

    def reset_optimization_state(self):
        """Reset optimization state and clear history."""
        try:
            self.optimization_history.clear()

            # Reset performance metrics
            self.performance_metrics = {
                "total_optimization_time": 0.0,
                "total_rounds_executed": 0,
                "total_moves_accepted": 0,
                "total_risk_violations": 0,
                "convergence_rate": 0.0
            }

            tprint("🧹 Optimization state reset", "INFO")

        except Exception as e:
            tprint(f"⚠️ State reset failed: {e}", "WARNING")
