"""
Clustering Orchestrator for NAS-TAS Clustering.

This module orchestrates the entire clustering pipeline by coordinating
the sequential steps and iterative optimization processes.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from .step1_feature_preparation import ClusteringContext
from .clustering_service import ClusteringService
from .feature_service import FeatureService
from .optimization_service import OptimizationService
from .hardware_service import HardwareService
from ..shared_utils import get_logger

class ClusteringOrchestrator:
    """Orchestrates the entire NAS-TAS clustering pipeline."""

    def __init__(self, verbose: bool = True):
        """Initialize the clustering orchestrator."""
        self.verbose = verbose
        self.logger = get_logger('ClusteringOrchestrator')

        # Initialize service layer components
        self.clustering_service = ClusteringService(verbose=verbose)
        self.feature_service = FeatureService(verbose=verbose)
        self.optimization_service = OptimizationService(verbose=verbose)
        self.hardware_service = HardwareService(verbose=verbose)

        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "step_times": {},
            "memory_usage": [],
            "error_count": 0,
            "success_count": 0
        }

    async def execute_clustering_pipeline(
        self,
        features: np.ndarray,
        market_data: pd.DataFrame,
        config: Any
    ) -> Dict[str, Any]:
        """Execute the complete clustering pipeline."""
        try:
            # Initialize performance tracking
            self.performance_metrics["start_time"] = time.time()
            tprint("🚀 Starting NAS-TAS Clustering Pipeline (Refactored)", "INFO")
            tprint("🎯 Using advanced 3-step iterative clustering with risk mitigation", "INFO")

            # Create clustering context
            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                original_feature_names=getattr(config, 'feature_names', None),
                feature_scores=getattr(config, 'feature_scores', {})
            )

            # Validate context before pipeline execution
            if not hasattr(context, 'original_features') or context.original_features is None:
                raise ValueError("Original features are None or not available in context")
            if not hasattr(context, 'market_data') or context.market_data is None or context.market_data.empty:
                raise ValueError("Market data is None or empty in context")

            # Execute pipeline steps
            context = await self._execute_pipeline_steps(context, config)

            # Finalize performance tracking
            self.performance_metrics["end_time"] = time.time()
            total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]

            tprint(f"🎉 NAS-TAS Clustering Pipeline completed in {total_time:.2f} seconds", "SUCCESS")

            # Add performance metrics to results
            if hasattr(context, 'final_results'):
                context.final_results['performance_metrics'] = self.performance_metrics
                context.final_results['clustering_method'] = 'advanced_3_step_iterative'
                context.final_results['risk_mitigation_enabled'] = True

            return context.final_results

        except Exception as e:
            self.performance_metrics["error_count"] += 1
            tprint(f"❌ Clustering pipeline failed: {e}", "ERROR")
            raise ValueError(f"Clustering pipeline failed: {e}")

    async def _execute_pipeline_steps(
        self,
        context: ClusteringContext,
        config: Any
    ) -> ClusteringContext:
        """Execute all pipeline steps in sequence."""
        try:
            # Step 1: Feature Preparation (using FeatureService)
            tprint("📊 Step 1: Feature Preparation", "INFO")
            step_start = time.time()
            feature_result = await self.feature_service.prepare_features_for_clustering(
                context.market_data, config
            )

            # Validate feature result
            if feature_result is None or len(feature_result) < 2:
                raise ValueError("Feature preparation returned None or insufficient results")

            features, feature_names, metadata = feature_result

            if features is None or features.size == 0:
                raise ValueError("Feature preparation returned None or empty features")

            if not hasattr(features, 'shape') or len(features.shape) != 2:
                raise ValueError(f"Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}")

            if feature_names is None:
                feature_names = []

            context.optimized_features = features
            context.optimized_feature_names = feature_names
            # feature_scores are available in the FeaturePreparationResult if needed
            context.feature_scores = {}
            self._record_step_time("step1_feature_preparation", time.time() - step_start)

            # Step 2: Initial Clustering (using ClusteringService)
            tprint("🔍 Step 2: Initial Clustering", "INFO")
            step_start = time.time()
            initial_assignments, optimal_k = await self.clustering_service.run_initial_clustering_only(
                context.optimized_features, context.market_data, config
            )
            context.initial_assignments = initial_assignments
            context.optimal_k = optimal_k
            self._record_step_time("step2_initial_clustering", time.time() - step_start)

            # Step 3: Iterative Optimization (using OptimizationService)
            tprint("🔄 Step 3: Iterative Optimization", "INFO")
            step_start = time.time()
            optimization_result = await self.optimization_service.run_optimization(
                context, config, max_iterations=100
            )
            context.optimized_assignments = optimization_result.final_context.optimized_assignments
            context = optimization_result.final_context
            self._record_step_time("iterative_optimization", time.time() - step_start)

            # Step 4: Validation (using ClusteringService)
            tprint("✅ Step 4: Validation", "INFO")
            step_start = time.time()
            validation_results = self.clustering_service.validate_clustering_quality(
                context.optimized_assignments, context.optimized_features
            )
            context.validation_results = validation_results
            self._record_step_time("step8_validation", time.time() - step_start)

            # Step 5: Results Consolidation (using ClusteringService)
            tprint("📋 Step 5: Results Consolidation", "INFO")
            step_start = time.time()
            final_results = {
                'cluster_assignments': context.optimized_assignments,
                'n_clusters': context.optimal_k,
                'optimization_history': optimization_result.optimization_history,
                'validation_results': validation_results,
                'performance_metrics': {
                    **self.performance_metrics,
                    **optimization_result.performance_metrics
                }
            }
            self._record_step_time("step9_results_consolidation", time.time() - step_start)

            # Step 6: Comprehensive Reporting (using ClusteringService)
            tprint("📊 Step 6: Comprehensive Reporting", "INFO")
            step_start = time.time()
            comprehensive_report = {
                'summary': f'Clustering completed with {context.optimal_k} clusters',
                'execution_time': optimization_result.total_execution_time,
                'convergence_status': optimization_result.convergence_status,
                'risk_violations': optimization_result.risk_violations
            }
            self._record_step_time("step10_comprehensive_reporting", time.time() - step_start)

            # Add comprehensive report to final results
            final_results['comprehensive_report'] = comprehensive_report

            # Store final results in context
            context.final_results = final_results
            self.performance_metrics["success_count"] += 1

            return context

        except Exception as e:
            self.performance_metrics["error_count"] += 1
            tprint(f"Pipeline execution failed: {e}", "ERROR")
            raise

    def _record_step_time(self, step_name: str, duration: float) -> None:
        """Record execution time for a step."""
        try:
            self.performance_metrics["step_times"][step_name] = duration
            tprint(f"⏱️ {step_name}: {duration:.2f}s", "INFO")
        except Exception as e:
            tprint(f"Failed to record step time for {step_name}: {e}", "WARNING")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the clustering pipeline."""
        try:
            if self.performance_metrics["start_time"] and self.performance_metrics["end_time"]:
                total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
            else:
                total_time = 0.0

            step_times = self.performance_metrics["step_times"]

            summary = {
                "total_execution_time": total_time,
                "step_breakdown": step_times,
                "success_count": self.performance_metrics["success_count"],
                "error_count": self.performance_metrics["error_count"],
                "average_step_time": np.mean(list(step_times.values())) if step_times else 0.0,
                "slowest_step": max(step_times.items(), key=lambda x: x[1])[0] if step_times else None,
                "fastest_step": min(step_times.items(), key=lambda x: x[1])[0] if step_times else None
            }

            return summary

        except Exception as e:
            tprint(f"Performance summary generation failed: {e}", "ERROR")
            return {"error": str(e)}

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        try:
            self.performance_metrics = {
                "start_time": None,
                "end_time": None,
                "step_times": {},
                "memory_usage": [],
                "error_count": 0,
                "success_count": 0
            }
            tprint("Performance metrics reset", "INFO")
        except Exception as e:
            tprint(f"Performance metrics reset failed: {e}", "ERROR")

    async def execute_step_individually(
        self,
        step_name: str,
        context: ClusteringContext,
        config: Any
    ) -> ClusteringContext:
        """Execute a single step individually for testing/debugging."""
        try:
            tprint(f"Executing individual step: {step_name}", "INFO")

            if step_name == "step1_feature_preparation":
                # Use FeatureService for feature preparation
                feature_result = await self.feature_service.prepare_features_for_clustering(
                    context.market_data, config
                )
                context.optimized_features = feature_result[0]
                context.optimized_feature_names = feature_result[1]
                context.feature_scores = {}
                return context

            elif step_name == "step2_initial_clustering":
                # Use ClusteringService for initial clustering
                initial_assignments, optimal_k = await self.clustering_service.run_initial_clustering_only(
                    context.optimized_features, context.market_data, config
                )
                context.initial_assignments = initial_assignments
                context.optimal_k = optimal_k
                return context

            elif step_name == "iterative_optimization":
                # Use OptimizationService for iterative optimization
                optimization_result = await self.optimization_service.run_optimization(
                    context, config, max_iterations=100
                )
                context.optimized_assignments = optimization_result.final_context.optimized_assignments
                context = optimization_result.final_context
                return context

            elif step_name == "step8_validation":
                # Use ClusteringService for validation
                validation_results = self.clustering_service.validate_clustering_quality(
                    context.optimized_assignments, context.optimized_features
                )
                context.validation_results = validation_results
                return context

            elif step_name == "step9_results_consolidation":
                # Use ClusteringService for results consolidation
                final_results = {
                    'cluster_assignments': context.optimized_assignments,
                    'n_clusters': context.optimal_k,
                    'validation_results': context.validation_results
                }
                context.final_results = final_results
                return context

            else:
                raise ValueError(f"Unknown step: {step_name}")

        except Exception as e:
            tprint(f"Individual step execution failed for {step_name}: {e}", "ERROR")
            raise

    def get_step_info(self) -> Dict[str, str]:
        """Get information about available steps."""
        return {
            "step1_feature_preparation": "Feature preparation and optimization",
            "step2_initial_clustering": "Initial clustering setup and regime assignment",
            "iterative_optimization": "Iterative optimization loop with convergence",
            "step8_validation": "Clustering validation and robustness testing",
            "step9_results_consolidation": "Results consolidation and artifact creation"
        }

    def validate_pipeline_requirements(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate that pipeline requirements are met."""
        try:
            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            # Check features
            if features is None or features.size == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("Features are empty or None")

            if features.shape[0] < 10:
                validation_results["warnings"].append("Very few samples for clustering")

            if features.shape[1] < 2:
                validation_results["valid"] = False
                validation_results["issues"].append("Insufficient features for clustering")

            # Check market data
            if market_data is None or market_data.empty:
                validation_results["warnings"].append("Market data is empty")

            # Check for NaN values
            if np.any(np.isnan(features)):
                validation_results["warnings"].append("Features contain NaN values")

            # Check for infinite values
            if np.any(np.isinf(features)):
                validation_results["warnings"].append("Features contain infinite values")

            return validation_results

        except Exception as e:
            tprint(f"Pipeline validation failed: {e}", "ERROR")
            return {"valid": False, "issues": [f"Validation error: {e}"], "warnings": []}
