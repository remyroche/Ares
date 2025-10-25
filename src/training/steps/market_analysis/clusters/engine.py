"""
Core Clustering Engine for HDBSCAN Clustering.

This module provides the main clustering engine that wraps initial clustering
and iterative optimization loops, orchestrating the three main steps.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from datetime import datetime

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, cleanup_m1_optimizers
)
from src.utils.common_utilities import (
    calculate_data_quality_metrics, safe_dataframe_operation,
    validate_dataframe_columns, create_summary_statistics
)
from src.utils.math_validation import (
    safe_divide, validate_finite, safe_log, safe_sqrt
)
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

from ..shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext
from .step2_initial_clustering import InitialClusteringStep
from .iterative_optimization import IterativeOptimization, ClusteringStats
from .optimizer import ClusteringOptimizer
from .metrics import ClusteringMetrics

@dataclass
class EngineConfig:
    """Configuration for the clustering engine."""
    # Core parameters
    max_iterations: int = 100
    convergence_tolerance: float = 1e-5
    enable_early_stopping: bool = True

    # Step orchestration
    run_step1: bool = True
    run_step2: bool = True
    run_step3: bool = True

    # Optimization settings
    optimization_method: str = "advanced_iterative"  # "advanced_iterative" or "basic"

    # Risk mitigation
    enable_risk_mitigation: bool = True
    risk_config: Optional[Dict[str, Any]] = None

    # Performance settings
    enable_parallel_processing: bool = False
    memory_optimization: bool = True

    # Logging and reporting
    verbose: bool = True
    generate_detailed_reports: bool = True

class ClusteringEngine:
    """
    Core clustering engine that orchestrates the entire clustering pipeline.

    This class wraps initial clustering and iterative optimization loops,
    coordinating Step 1 (local), Step 2 (global), and Step 3 (split).
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the clustering engine."""
        self.config = config or EngineConfig()
        self.logger = get_logger('ClusteringEngine')

        # Initialize components
        self.step1 = InitialClusteringStep(verbose=self.config.verbose)
        self.step2 = None  # Will be initialized based on optimization method
        self.step3 = None  # Will be initialized based on optimization method

        # Hardware service integration
        try:
            from .hardware_service import HardwareService
            self.hardware_service = HardwareService(verbose=self.config.verbose)
            self.hardware_integration_enabled = True
        except ImportError:
            self.hardware_service = None
            self.hardware_integration_enabled = False

        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "step_times": {},
            "memory_usage": [],
            "convergence_info": {},
            "iteration_count": 0,
            "hardware_accelerations": 0,
            "memory_optimizations": 0
        }

        # Results storage
        self.results = {}

    def _initialize_hardware_optimizations(self) -> None:
        """Initialize hardware optimizations for M1 chips."""
        try:
            # Initialize matrix operations with hardware acceleration
            self.matrix_ops = UnifiedMatrixOperations()

            # Get hardware managers
            self.hardware_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()

            if self.hardware_manager or self.memory_optimizer:
                tprint("🖥️ Hardware optimizations initialized for clustering engine", "INFO")
            else:
                tprint("⚠️ Hardware optimizations not available, using CPU fallback", "WARNING")

        except Exception as e:
            tprint(f"❌ Hardware initialization failed: {e}", "ERROR")
            self.hardware_manager = None
            self.memory_optimizer = None
            self.matrix_ops = None

    async def execute_clustering(
        self,
        features: np.ndarray,
        market_data: pd.DataFrame,
        config: Any,
        context: Optional[ClusteringContext] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete clustering pipeline.

        Args:
            features: Feature matrix for clustering
            market_data: Market data DataFrame
            config: Configuration object
            context: Optional clustering context

        Returns:
            Dictionary containing clustering results
        """
        try:
            # Initialize performance tracking
            self.performance_metrics["start_time"] = time.time()
            tprint("🚀 Starting Clustering Engine Execution", "INFO")

            # Apply hardware optimizations if available
            optimization_context = None
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    optimization_context = self.hardware_service.get_optimization_context()
                    optimization_context.__enter__()
                    tprint("🧠 Hardware optimizations activated", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware optimization context failed: {e}", "WARNING")

            try:
                # Create or validate context
                if context is None:
                    context = ClusteringContext(
                        original_features=features,
                        market_data=market_data,
                        original_feature_names=getattr(config, 'feature_names', None),
                        feature_scores=getattr(config, 'feature_scores', {})
                    )

                # Execute pipeline steps
                context = await self._execute_pipeline_steps(context, config)

            finally:
                # Clean up hardware optimization context
                if optimization_context:
                    try:
                        optimization_context.__exit__(None, None, None)
                        tprint("🧠 Hardware optimizations deactivated", "SUCCESS")
                    except Exception as e:
                        tprint(f"⚠️ Hardware optimization context cleanup failed: {e}", "WARNING")

            # Finalize performance tracking
            self.performance_metrics["end_time"] = time.time()
            total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]

            tprint(f"🎉 Clustering Engine completed in {total_time:.2f} seconds", "SUCCESS")

            # Prepare final results
            results = self._prepare_final_results(context, config)

            return results

        except Exception as e:
            tprint(f"❌ Clustering engine execution failed: {e}", "ERROR")
            raise ValueError(f"Clustering engine execution failed: {e}")

    async def _execute_pipeline_steps(
        self,
        context: ClusteringContext,
        config: Any
    ) -> ClusteringContext:
        """Execute all pipeline steps in sequence."""

        try:
            # Step 1: Initial clustering setup (if enabled)
            if self.config.run_step1:
                tprint("📊 Step 1: Initial Clustering Setup", "INFO")
                step_start = time.time()
                context = await self.step1.execute(context, config)
                self._record_step_time("step1_initial_clustering", time.time() - step_start)

            # Step 2: Optimization (based on method)
            if self.config.run_step2:
                tprint(f"🔍 Step 2: {self.config.optimization_method.title()} Optimization", "INFO")
                step_start = time.time()

                if self.config.optimization_method == "advanced_iterative":
                    # Use the existing iterative optimization
                    k = context.optimal_k if context.optimal_k is not None else 5
                    iterative_optimizer = IterativeOptimization(verbose=self.config.verbose, k=k)
                    context = await iterative_optimizer.execute_optimization_loop(
                        context, config, max_iterations=self.config.max_iterations
                    )
                else:
                    # Use the new optimizer component
                    if self.step2 is None:
                        self.step2 = ClusteringOptimizer(verbose=self.config.verbose)
                    context = await self.step2.execute_optimization(context, config)

                self._record_step_time("step2_optimization", time.time() - step_start)

            # Step 3: Final refinement (if enabled)
            if self.config.run_step3:
                tprint("🔧 Step 3: Final Refinement", "INFO")
                step_start = time.time()

                # Apply final metrics calculation and validation
                metrics_calculator = ClusteringMetrics(verbose=self.config.verbose)
                context = await metrics_calculator.compute_all_metrics(context, config)

                self._record_step_time("step3_refinement", time.time() - step_start)

            return context

        except Exception as e:
            tprint(f"Pipeline step execution failed: {e}", "ERROR")
            raise

    def _record_step_time(self, step_name: str, duration: float) -> None:
        """Record execution time for a step."""
        try:
            self.performance_metrics["step_times"][step_name] = duration
            tprint(f"⏱️ {step_name}: {duration:.2f}s", "INFO")
        except Exception as e:
            tprint(f"Failed to record step time for {step_name}: {e}", "WARNING")

    def _prepare_final_results(
        self,
        context: ClusteringContext,
        config: Any
    ) -> Dict[str, Any]:
        """Prepare final clustering results."""

        try:
            # Core results
            results = {
                "assignments": getattr(context, 'optimized_assignments', None),
                "n_clusters": getattr(context, 'final_k', None),
                "cluster_centers": None,  # Would be computed if needed
                "performance_metrics": self.performance_metrics,
                "engine_config": self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
                "execution_timestamp": datetime.now().isoformat()
            }

            # Add clustering statistics if available
            if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                stats = ClusteringStats(context.optimized_features, context.optimized_assignments)

                results.update({
                    "cv_ratio": stats.get_cv_ratio(),
                    "balance_score": stats.get_balance_score(),
                    "objective_value": stats.get_objective_value(),
                    "cluster_sizes": stats.cluster_sizes.tolist(),
                    "total_samples": stats.n_samples,
                    "total_features": stats.n_features
                })

            # Add detailed reports if requested
            if self.config.generate_detailed_reports:
                results["detailed_reports"] = self._generate_detailed_reports(context, config)

            return results

        except Exception as e:
            tprint(f"Final results preparation failed: {e}", "ERROR")
            return {"error": str(e)}

    def _generate_detailed_reports(
        self,
        context: ClusteringContext,
        config: Any
    ) -> Dict[str, Any]:
        """Generate detailed reports for the clustering results."""

        try:
            reports = {}

            # Performance report
            reports["performance"] = self._generate_performance_report()

            # Quality metrics report
            if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                metrics_calculator = ClusteringMetrics(verbose=False)
                reports["quality_metrics"] = metrics_calculator.generate_quality_report(
                    context.optimized_features, context.optimized_assignments
                )

            # Convergence report
            if self.performance_metrics.get("convergence_info"):
                reports["convergence"] = self.performance_metrics["convergence_info"]

            return reports

        except Exception as e:
            tprint(f"Detailed reports generation failed: {e}", "ERROR")
            return {"error": str(e)}

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance summary report."""

        try:
            if self.performance_metrics["start_time"] and self.performance_metrics["end_time"]:
                total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
            else:
                total_time = 0.0

            step_times = self.performance_metrics["step_times"]

            return {
                "total_execution_time": total_time,
                "step_breakdown": step_times,
                "average_step_time": np.mean(list(step_times.values())) if step_times else 0.0,
                "slowest_step": max(step_times.items(), key=lambda x: x[1])[0] if step_times else None,
                "fastest_step": min(step_times.items(), key=lambda x: x[1])[0] if step_times else None,
                "memory_peak_usage": max(self.performance_metrics["memory_usage"]) if self.performance_metrics["memory_usage"] else 0
            }

        except Exception as e:
            tprint(f"Performance report generation failed: {e}", "ERROR")
            return {"error": str(e)}

    async def execute_step_individually(
        self,
        step_name: str,
        features: np.ndarray,
        market_data: pd.DataFrame,
        config: Any,
        context: Optional[ClusteringContext] = None
    ) -> Dict[str, Any]:
        """Execute a single step individually for testing/debugging."""

        try:
            tprint(f"Executing individual step: {step_name}", "INFO")

            # Create context if not provided
            if context is None:
                context = ClusteringContext(
                    original_features=features,
                    market_data=market_data,
                    original_feature_names=getattr(config, 'feature_names', None),
                    feature_scores=getattr(config, 'feature_scores', {})
                )

            if step_name == "step1_initial_clustering":
                context = await self.step1.execute(context, config)
                return {"context": context, "assignments": getattr(context, 'initial_assignments', None)}

            elif step_name == "step2_optimization":
                if self.config.optimization_method == "advanced_iterative":
                    k = context.optimal_k if context.optimal_k is not None else 5
                    iterative_optimizer = IterativeOptimization(verbose=self.config.verbose, k=k)
                    context = await iterative_optimizer.execute_optimization_loop(
                        context, config, max_iterations=self.config.max_iterations
                    )
                else:
                    if self.step2 is None:
                        self.step2 = ClusteringOptimizer(verbose=self.config.verbose)
                    context = await self.step2.execute_optimization(context, config)

                return {"context": context, "assignments": getattr(context, 'optimized_assignments', None)}

            elif step_name == "step3_metrics":
                metrics_calculator = ClusteringMetrics(verbose=self.config.verbose)
                context = await metrics_calculator.compute_all_metrics(context, config)
                return {"context": context, "metrics": getattr(context, 'clustering_metrics', {})}

            else:
                raise ValueError(f"Unknown step: {step_name}")

        except Exception as e:
            tprint(f"Individual step execution failed for {step_name}: {e}", "ERROR")
            raise

    def get_engine_info(self) -> Dict[str, str]:
        """Get information about the clustering engine."""
        return {
            "engine_type": "ClusteringEngine",
            "optimization_method": self.config.optimization_method,
            "supports_risk_mitigation": self.config.enable_risk_mitigation,
            "supports_early_stopping": self.config.enable_early_stopping,
            "supports_parallel_processing": self.config.enable_parallel_processing,
            "steps_enabled": {
                "step1": self.config.run_step1,
                "step2": self.config.run_step2,
                "step3": self.config.run_step3
            }
        }

    def reset_engine(self) -> None:
        """Reset the engine state."""
        try:
            self.performance_metrics = {
                "start_time": None,
                "end_time": None,
                "step_times": {},
                "memory_usage": [],
                "convergence_info": {},
                "iteration_count": 0
            }
            self.results = {}
            tprint("Clustering engine reset", "INFO")
        except Exception as e:
            tprint(f"Engine reset failed: {e}", "ERROR")

    def validate_requirements(
        self,
        features: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate that engine requirements are met using enhanced utilities."""

        try:
            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "data_quality": {}
            }

            # Validate features using math utilities
            if not validate_finite(features, "features"):
                validation_results["valid"] = False
                validation_results["issues"].append("Features contain non-finite values")

            if features is None or features.size == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("Features are empty or None")

            if features.shape[0] < 10:
                validation_results["warnings"].append("Very few samples for clustering")

            if features.shape[1] < 2:
                validation_results["valid"] = False
                validation_results["issues"].append("Insufficient features for clustering")

            # Validate market data using common utilities
            if market_data is not None and not len(market_data) == 0:
                # Calculate data quality metrics
                try:
                    data_quality = calculate_data_quality_metrics(market_data)
                    validation_results["data_quality"] = data_quality

                    # Check for required columns
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if not validate_dataframe_columns(market_data, required_columns):
                        validation_results["warnings"].append("Market data missing some recommended columns")
                except Exception as e:
                    validation_results["warnings"].append(f"Data quality check failed: {e}")
            else:
                validation_results["warnings"].append("Market data is empty")

            # Enhanced validation using matrix operations
            if self.matrix_ops and features is not None:
                try:
                    # Check for numerical stability
                    feature_norms = self.matrix_ops.compute_norms(features)
                    if np.any(feature_norms == 0):
                        validation_results["warnings"].append("Some features have zero variance")
                except Exception as e:
                    validation_results["warnings"].append(f"Matrix validation failed: {e}")

            return validation_results

        except Exception as e:
            tprint(f"Engine validation failed: {e}", "ERROR")
            return {"valid": False, "issues": [f"Validation error: {e}"], "warnings": []}
