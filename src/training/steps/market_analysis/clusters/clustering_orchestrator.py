"""
Clustering Orchestrator for NAS-TAS Clustering.

This module orchestrates the entire clustering pipeline by coordinating
the sequential steps and iterative optimization processes.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Awaitable
from dataclasses import dataclass, field
import time
from datetime import datetime

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance,
    tprint_data_preview, LogLevel
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
    validate_finite, validate_array_finite, safe_divide, safe_log, safe_sqrt,
    safe_power, safe_correlation, safe_mean, safe_std, validate_positive
)
from src.utils.data.unified_data_utils import (
    UnifiedDataUtils, DataQualityMetrics, DataValidationResult
)
from src.utils.data.quality.comprehensive_quality_scorer import (
    ComprehensiveQualityScorer, QualityScoreConfig
)
from src.utils.kline_parquet import KlinesParquetManager, KlinesMetadata
from src.utils.artifact_manager import ArtifactManager, ArtifactConfig
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, IntegratedHardwareManager
)
from src.utils.hardware.enhanced_caching_system import (
    EnhancedCachingSystem, CacheConfig
)
from src.utils.hardware.advanced_memory_manager import (
    AdvancedMemoryManager, MemoryConfig
)

from .step1_feature_preparation import ClusteringContext
from .clustering_service import ClusteringService
from .feature_service import FeatureService
from .optimization_service import OptimizationService
from .hardware_service import HardwareService
from .shared_utils import get_logger

class ClusteringOrchestrator:
    """Orchestrates the entire NAS-TAS clustering pipeline."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the clustering orchestrator."""
        tprint("🚀 Initializing ClusteringOrchestrator", "INFO")
        self.verbose = verbose
        self.logger = get_logger('ClusteringOrchestrator')
        tprint_debug(f"Orchestrator verbose mode: {verbose}")

        # Initialize service layer components
        tprint("🔧 Initializing service layer components", "INFO")
        self.clustering_service = ClusteringService(verbose=verbose)
        tprint_debug("ClusteringService initialized")
        self.feature_service = FeatureService(verbose=verbose)
        tprint_debug("FeatureService initialized")
        self.optimization_service = OptimizationService(verbose=verbose)
        tprint_debug("OptimizationService initialized")
        self.hardware_service = HardwareService(verbose=verbose)
        tprint_debug("HardwareService initialized")

        # Performance tracking with enhanced metrics
        self.performance_metrics: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "step_times": {},
            "memory_usage": [],
            "error_count": 0,
            "success_count": 0,
            "data_quality_metrics": {},
            "optimization_stats": {}
        }
        
        # Initialize memory monitoring
        tprint("🧠 Initializing memory monitoring", "INFO")
        self.initial_memory = get_memory_usage()
        tprint_debug(f"Orchestrator initialized - Initial memory: {self.initial_memory['rss']:.1f}MB")
        
        # Initialize data utilities
        tprint("📊 Initializing data utilities", "INFO")
        self.data_utils = UnifiedDataUtils()
        tprint_debug("UnifiedDataUtils initialized")
        self.quality_scorer = ComprehensiveQualityScorer(QualityScoreConfig())
        tprint_debug("ComprehensiveQualityScorer initialized")
        self.klines_manager = KlinesParquetManager()
        tprint_debug("KlinesParquetManager initialized")
        self.artifact_manager = ArtifactManager(ArtifactConfig())
        tprint_debug("ArtifactManager initialized")
        
        # Initialize hardware utilities
        tprint("⚡ Initializing hardware utilities", "INFO")
        try:
            self.hardware_manager = get_integrated_hardware_manager()
            tprint_debug("Integrated hardware manager initialized")
            self.caching_system = EnhancedCachingSystem(CacheConfig())
            tprint_debug("Enhanced caching system initialized")
            self.memory_manager = AdvancedMemoryManager(MemoryConfig())
            tprint_debug("Advanced memory manager initialized")
            tprint("🔧 Hardware utilities initialized", "SUCCESS")
        except Exception as e:
            tprint(f"⚠️ Failed to initialize hardware utilities: {e}", "WARNING")
            tprint_debug(f"Hardware initialization error details: {e}")
            self.hardware_manager = None
            self.caching_system = None
            self.memory_manager = None

    async def execute_clustering_pipeline(
        self,
        features: np.ndarray,
        market_data: pd.DataFrame,
        config: Any
    ) -> Dict[str, Any]:
        """Execute the complete clustering pipeline with enhanced monitoring."""
        try:
            # Initialize performance tracking
            self.performance_metrics["start_time"] = time.time()
            tprint("🚀 Starting NAS-TAS Clustering Pipeline (Refactored)", "INFO")
            tprint("🎯 Using advanced 3-step iterative clustering with risk mitigation", "INFO")
            tprint_debug(f"Input features shape: {features.shape}")
            tprint_debug(f"Market data shape: {market_data.shape}")
            tprint_debug(f"Config type: {type(config)}")

            # Add data preview logging for troubleshooting
            tprint_data_preview(features, "orchestrator_input_features", max_rows=5, max_cols=10, level=LogLevel.DEBUG)
            tprint_data_preview(market_data, "orchestrator_input_market_data", max_rows=5, max_cols=10, level=LogLevel.DEBUG)

            # Validate input data quality
            tprint("🔍 Validating input data quality", "INFO")
            with memory_monitor("Data Validation"):
                data_quality = calculate_data_quality_metrics(market_data)
                self.performance_metrics["data_quality_metrics"] = data_quality
                tprint_debug(f"Data quality metrics: {data_quality}")
                tprint_info(f"Data quality score: {data_quality.get('overall_score', 'N/A')}")
                
                # Analyze NaN values if present
                if data_quality.get('missing_percentage', 0) > 0:
                    tprint_warning(f"Missing data detected: {data_quality.get('missing_percentage', 0):.2f}%")
                    nan_analysis = analyze_nan_values_detailed(market_data)
                    tprint_warning(format_nan_analysis_report(nan_analysis, "⚠️ "))
                else:
                    tprint("✅ No missing data detected", "SUCCESS")

            # Create clustering context
            tprint("📋 Creating clustering context", "INFO")
            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                original_feature_names=getattr(config, 'feature_names', None),
                feature_scores=getattr(config, 'feature_scores', {})
            )
            tprint_debug(f"Context created with {len(context.original_feature_names or [])} feature names")
            tprint_debug(f"Feature scores available: {len(context.feature_scores or {})}")

            # Validate context before pipeline execution with enhanced validation
            if not hasattr(context, 'original_features') or context.original_features is None:
                raise ValueError("Original features are None or not available in context")
            if not hasattr(context, 'market_data') or context.market_data is None or context.market_data.empty:
                raise ValueError("Market data is None or empty in context")
            
            # Validate finite values in features
            try:
                validate_array_finite(features, "original_features")
            except ValueError as e:
                tprint_warning(f"Feature validation warning: {e}")
                # Continue with warning rather than failing

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

            # Add data preview logging for final results
            if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                tprint_data_preview(context.optimized_assignments, "orchestrator_final_assignments", max_rows=10, level=LogLevel.DEBUG)
            if hasattr(context, 'final_results') and context.final_results is not None:
                tprint_data_preview(context.final_results, "orchestrator_final_results", level=LogLevel.DEBUG)

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
