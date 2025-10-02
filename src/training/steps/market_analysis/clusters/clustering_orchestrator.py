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

from .step1_feature_preparation import FeaturePreparationStep, ClusteringContext
from .step2_initial_clustering import InitialClusteringStep
from .iterative_optimization import IterativeOptimization
from .step8_validation import ValidationStep
from .step9_results_consolidation import ResultsConsolidationStep
from ...shared_utils import get_logger


class ClusteringOrchestrator:
    """Orchestrates the entire NAS-TAS clustering pipeline."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the clustering orchestrator."""
        self.verbose = verbose
        self.logger = get_logger('ClusteringOrchestrator')
        
        # Initialize all steps
        self.step1 = FeaturePreparationStep(verbose=verbose)
        self.step2 = InitialClusteringStep(verbose=verbose)
        self.iterative_optimizer = IterativeOptimization(verbose=verbose)
        self.step8 = ValidationStep(verbose=verbose)
        self.step9 = ResultsConsolidationStep(verbose=verbose)
        
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
            
            # Create clustering context
            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                original_feature_names=getattr(config, 'feature_names', None),
                feature_scores=getattr(config, 'feature_scores', {})
            )
            
            # Execute pipeline steps
            context = await self._execute_pipeline_steps(context, config)
            
            # Finalize performance tracking
            self.performance_metrics["end_time"] = time.time()
            total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
            
            tprint(f"🎉 NAS-TAS Clustering Pipeline completed in {total_time:.2f} seconds", "SUCCESS")
            
            # Add performance metrics to results
            if hasattr(context, 'final_results'):
                context.final_results['performance_metrics'] = self.performance_metrics
            
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
            # Step 1: Feature Preparation
            tprint("📊 Step 1: Feature Preparation", "INFO")
            step_start = time.time()
            context = await self.step1.execute(context, config)
            self._record_step_time("step1_feature_preparation", time.time() - step_start)
            
            # Step 2: Initial Clustering
            tprint("🔍 Step 2: Initial Clustering", "INFO")
            step_start = time.time()
            context = await self.step2.execute(context, config)
            self._record_step_time("step2_initial_clustering", time.time() - step_start)
            
            # Iterative Optimization Loop
            tprint("🔄 Iterative Optimization Loop", "INFO")
            step_start = time.time()
            context = await self.iterative_optimizer.execute_optimization_loop(
                context, config, max_iterations=100
            )
            self._record_step_time("iterative_optimization", time.time() - step_start)
            
            # Step 8: Validation
            tprint("✅ Step 8: Validation", "INFO")
            step_start = time.time()
            context = await self.step8.execute(context, config)
            self._record_step_time("step8_validation", time.time() - step_start)
            
            # Step 9: Results Consolidation
            tprint("📋 Step 9: Results Consolidation", "INFO")
            step_start = time.time()
            final_results = await self.step9.execute(context, config)
            self._record_step_time("step9_results_consolidation", time.time() - step_start)
            
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
                return await self.step1.execute(context, config)
            elif step_name == "step2_initial_clustering":
                return await self.step2.execute(context, config)
            elif step_name == "iterative_optimization":
                return await self.iterative_optimizer.execute_optimization_loop(context, config)
            elif step_name == "step8_validation":
                return await self.step8.execute(context, config)
            elif step_name == "step9_results_consolidation":
                final_results = await self.step9.execute(context, config)
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