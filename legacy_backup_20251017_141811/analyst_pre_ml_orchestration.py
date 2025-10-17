"""
Analyst Pre-ML Orchestration - 60m Timeframe Feature Engineering

This orchestrator applies the complete pre-training pipeline for Analyst models:
1. Multi-horizon profit labeling with differentiated horizons
2. Feature lookback period optimization per regime/cluster
3. Interactive feature generation (interaction, polynomial, cross-timeframe)
4. Final feature selection (multi-stage: 120→100→80→60)

ANALYST CONFIGURATION:
- Timeframe: 60m (higher timeframe for strategic IF-to-trade decisions)
- Training Data: ALL market data (not filtered)
- Output: Features optimized for Analyst model training
- Per-regime optimization: Yes, using regime assignments from market_analysis
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import traceback
import asyncio
import time
from contextlib import contextmanager
from pathlib import Path

# Import pre-training sub-pipeline
try:
    from ..pre_training.sub_pipeline import (
        PreTrainingSubPipeline, SubPipelineConfig, SubPipelineResult, SubPipelineStatus,
        PipelineResultDict
    )
    PRE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import PreTrainingSubPipeline: {e}")
    PRE_TRAINING_AVAILABLE = False

# Import gate feature protection
try:
    from ..pre_training.gate_feature_integration import (
        GateFeaturePipelineManager, enable_gate_protection
    )
    GATE_PROTECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Gate feature protection not available: {e}")
    GATE_PROTECTION_AVAILABLE = False

# Enhanced imports with utility integration
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_timer
    )
    from src.utils.common_operations import (
        safe_divide, validate_dataframe, validate_dataframe_columns,
        optimize_dataframe_dtypes, calculate_data_quality_metrics,
        get_dataframe_info, safe_json_dump, safe_json_load,
        get_m1_memory_optimizer, get_m1_cpu_optimizer, get_m1_gpu_manager,
        integrate_with_m1_optimizers, cleanup_m1_optimizers
    )
    from src.utils.math_validation import (
        validate_finite, validate_positive, validate_range,
        validate_numeric_array, safe_log, safe_sqrt, safe_power
    )
    from src.utils.data.quality.data_quality import DataQualityFramework
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        OptimizationConfig, BayesianTPEOptimizer
    )
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    from src.utils.matrix_operations.hardware_integration import (
        HardwareOptimizedMatrixProcessor
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import utilities: {e}")
    UTILS_AVAILABLE = False

class OrchestrationPhase(Enum):
    """Orchestration execution phases."""
    HORIZON_LABELING = "horizon_labeling"
    LOOKBACK_OPTIMIZATION = "lookback_optimization"
    INTERACTIVE_FEATURE_GENERATION = "interactive_feature_generation"
    FEATURE_SELECTION = "feature_selection"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AnalystPreMLConfig:
    """Enhanced configuration for Analyst pre-ML orchestration with utility integration."""
    # Data configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "60m"  # ANALYST USES 60m TIMEFRAME
    data_dir: str = "historical_data"

    # Execution parameters
    enable_per_regime_optimization: bool = False  # Disabled - using regime probabilities as features instead
    enable_per_cluster_optimization: bool = True

    # Gate feature protection
    enable_gate_protection: bool = True
    gate_protection_config: Optional[Dict[str, Any]] = None

    # Output configuration
    output_directory: str = "generated/analyst_pre_ml"
    save_intermediate_results: bool = True

    # Hardware optimization
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0

    # Hardware optimization settings
    enable_hardware_optimization: bool = True
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive

    # Bayesian TPE optimization settings
    enable_bayesian_optimization: bool = True
    tpe_trials: int = 50
    tpe_timeout: Optional[float] = 300.0
    tpe_metric: str = "sharpe_ratio"

    # Data quality settings
    enable_data_quality_checks: bool = True
    data_quality_threshold: float = 0.8
    enable_outlier_detection: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval_seconds: float = 1.0
    enable_memory_monitoring: bool = True

    # Advanced validation settings
    enable_strict_validation: bool = True
    validation_tolerance: float = 1e-6

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not UTILS_AVAILABLE:
            return

        try:
            # Validate memory limit
            self.memory_limit_gb = validate_positive(self.memory_limit_gb, "memory_limit_gb")

            # Validate TPE settings
            if self.enable_bayesian_optimization:
                self.tpe_trials = validate_positive(self.tpe_trials, "tpe_trials")
                if self.tpe_timeout is not None:
                    self.tpe_timeout = validate_positive(self.tpe_timeout, "tpe_timeout")

            # Validate data quality threshold
            if self.enable_data_quality_checks:
                self.data_quality_threshold = validate_range(
                    self.data_quality_threshold, 0.0, 1.0, "data_quality_threshold"
                )

            # Validate monitoring interval
            if self.enable_performance_monitoring:
                self.monitoring_interval_seconds = validate_positive(
                    self.monitoring_interval_seconds, "monitoring_interval_seconds"
                )

            tprint_debug(f"✅ AnalystPreMLConfig validated successfully")

        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise

@dataclass
class AnalystPreMLResult:
    """Enhanced result of Analyst pre-ML orchestration with comprehensive metrics."""
    # Execution metadata
    success: bool = False
    execution_time: float = 0.0
    phase: OrchestrationPhase = OrchestrationPhase.HORIZON_LABELING

    # Step results
    horizon_labeling_result: Optional[Dict[str, Any]] = None
    lookback_optimization_result: Optional[Dict[str, Any]] = None
    interactive_feature_generation_result: Optional[Dict[str, Any]] = None
    feature_selection_result: Optional[Dict[str, Any]] = None

    # Output data
    final_features: Optional[pd.DataFrame] = None
    selected_feature_names: Optional[List[str]] = None

    # Enhanced metadata
    total_samples: int = 0
    total_features_generated: int = 0
    final_feature_count: int = 0
    error_message: Optional[str] = None

    # Performance metrics
    memory_usage_peak_mb: float = 0.0
    cpu_usage_peak_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0 if UTILS_AVAILABLE else 0.0

    # Data quality metrics
    data_quality_score: float = 0.0
    outlier_percentage: float = 0.0
    missing_data_percentage: float = 0.0

    # Step timing breakdown
    horizon_labeling_time: float = 0.0
    lookback_optimization_time: float = 0.0
    interactive_feature_generation_time: float = 0.0
    feature_selection_time: float = 0.0

    # Hardware optimization metrics
    hardware_optimization_enabled: bool = False
    m1_optimization_enabled: bool = False
    gpu_acceleration_used: bool = False
    parallel_processing_efficiency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'success': self.success,
            'execution_time': self.execution_time,
            'phase': self.phase.value,
            'total_samples': self.total_samples,
            'total_features_generated': self.total_features_generated,
            'final_feature_count': self.final_feature_count,
            'memory_usage_peak_mb': self.memory_usage_peak_mb,
            'cpu_usage_peak_percent': self.cpu_usage_peak_percent,
            'data_quality_score': self.data_quality_score,
            'outlier_percentage': self.outlier_percentage,
            'missing_data_percentage': self.missing_data_percentage,
            'hardware_optimization_enabled': self.hardware_optimization_enabled,
            'm1_optimization_enabled': self.m1_optimization_enabled,
            'gpu_acceleration_used': self.gpu_acceleration_used,
            'parallel_processing_efficiency': self.parallel_processing_efficiency,
            'error_message': self.error_message
        }

class AnalystPreMLOrchestrator:
    """
    Enhanced Analyst Pre-ML Orchestration with utility integration.

    Orchestrates the complete pre-training pipeline for Analyst models on 60m timeframe.
    Applies per-regime/cluster optimization for all feature engineering steps.
    Includes hardware optimization, performance monitoring, and advanced validation.
    """

    def __init__(self, config: Optional[AnalystPreMLConfig] = None):
        """Initialize the Analyst pre-ML orchestrator with enhanced capabilities."""
        try:
            self.config = config or AnalystPreMLConfig()
            self.logger = system_logger.getChild('AnalystPreMLOrchestrator')

            # Initialize performance monitoring
            self.performance_monitor = None
            self.hardware_manager = None
            self.memory_optimizer = None
            self.data_quality_checker = None

            # Initialize hardware optimization
            self._initialize_hardware_optimization()

            # Initialize data quality checker
            if UTILS_AVAILABLE and self.config.enable_data_quality_checks:
                try:
                    self.data_quality_checker = DataQualityFramework()
                    tprint_debug("✅ Data quality checker initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Data quality checker initialization failed: {e}")

            # Initialize Bayesian TPE optimizer for feature selection
            self.tpe_optimizer = None
            if UTILS_AVAILABLE and self.config.enable_bayesian_optimization:
                try:
                    self.tpe_optimizer = BayesianTPEOptimizer(
                        OptimizationConfig(
                            n_trials=self.config.tpe_trials,
                            timeout=self.config.tpe_timeout,
                            direction='maximize',
                            metric_name=self.config.tpe_metric,
                            enable_hardware_optimization=self.config.enable_hardware_optimization,
                            workload_type='ml_training',
                            optimization_level=self.config.optimization_level
                        )
                    )
                    tprint_debug("✅ Bayesian TPE optimizer initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ TPE optimizer initialization failed: {e}")

            # Initialize pre-training pipeline
            if PRE_TRAINING_AVAILABLE:
                self.pre_training_pipeline = PreTrainingSubPipeline()
                tprint_success("✅ Pre-training pipeline initialized for Analyst")
            else:
                self.pre_training_pipeline = None
                tprint_error("❌ Pre-training pipeline not available")

            tprint_success(f"✅ AnalystPreMLOrchestrator initialized (timeframe: {self.config.timeframe})")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystPreMLOrchestrator: {e}")
            raise

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        if not UTILS_AVAILABLE or not self.config.enable_hardware_optimization:
            return

        try:
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager.get_instance()
            self.hardware_manager.configure_workload(
                WorkloadType.ML_TRAINING,
                OptimizationLevel.BALANCED if self.config.optimization_level == "balanced" else
                OptimizationLevel.CONSERVATIVE if self.config.optimization_level == "conservative" else
                OptimizationLevel.AGGRESSIVE
            )

            # Initialize M1 optimizers if available and enabled
            if self.config.enable_m1_optimization:
                self.memory_optimizer = get_m1_memory_optimizer()
                if self.memory_optimizer:
                    self.memory_optimizer.start_monitoring()
                    tprint_debug("✅ M1 memory optimizer initialized")

            tprint_debug("✅ Hardware optimization initialized")

        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")

    def _cleanup_hardware_optimization(self):
        """Clean up hardware optimization resources."""
        if not UTILS_AVAILABLE:
            return

        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'stop_monitoring'):
                self.memory_optimizer.stop_monitoring()

            if self.hardware_manager:
                self.hardware_manager.cleanup()

            cleanup_m1_optimizers()
            tprint_debug("✅ Hardware optimization cleaned up")

        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization cleanup failed: {e}")

    @staticmethod
    def _extract_failure_details(step_result: SubPipelineResult) -> Tuple[Optional[str], str]:
        """Enhanced failure details extraction with better error context."""
        failure = getattr(step_result, 'failure', None)
        message = step_result.error_message or (failure.message if failure else 'Unknown error')
        error_code = getattr(step_result, 'error_code', None) or (failure.error_code if failure else None)
        return error_code, message

    async def _validate_input_data(self, training_data: pd.DataFrame, regime_assignments: Optional[pd.DataFrame] = None):
        """Validate input data quality and structure."""
        if not UTILS_AVAILABLE or not self.config.enable_strict_validation:
            return

        try:
            # Basic DataFrame validation
            if not validate_dataframe(training_data):
                raise ValueError("Input training_data is not a valid DataFrame")

            # Check for minimum required samples
            min_samples = 100  # Configurable minimum
            if len(training_data) < min_samples:
                tprint_warning(f"⚠️ Low sample count: {len(training_data)} < {min_samples}")

            # Validate essential columns (OHLCV + timestamp)
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(training_data, required_columns):
                tprint_warning(f"⚠️ Missing some expected columns: {required_columns}")

            # Data quality assessment
            if self.data_quality_checker:
                quality_report = self.data_quality_checker.analyze_dataframe(training_data)
                quality_score = quality_report.get('overall_score', 0.0)

                if quality_score < self.config.data_quality_threshold:
                    tprint_warning(f"⚠️ Low data quality score: {quality_score:.3f}")

                result = AnalystPreMLResult()
                result.data_quality_score = quality_score
                result.missing_data_percentage = quality_report.get('missing_percentage', 0.0)
                result.outlier_percentage = quality_report.get('outlier_percentage', 0.0)

            # Validate regime assignments if provided
            if regime_assignments is not None:
                if not validate_dataframe(regime_assignments):
                    raise ValueError("Regime assignments is not a valid DataFrame")

                # Check alignment with training data
                if len(regime_assignments) != len(training_data):
                    tprint_warning(f"⚠️ Regime assignments length mismatch: {len(regime_assignments)} vs {len(training_data)}")

            tprint_debug("✅ Input data validation completed")

        except Exception as e:
            tprint_error(f"❌ Input data validation failed: {e}")
            raise

    async def _start_performance_monitoring(self):
        """Start performance monitoring for the orchestration process."""
        if not UTILS_AVAILABLE or not self.config.enable_performance_monitoring:
            return None

        try:
            # Initialize performance tracking
            context = {
                'start_time': time.time(),
                'memory_baseline': self._get_memory_usage(),
                'cpu_baseline': self._get_cpu_usage()
            }

            tprint_debug("✅ Performance monitoring started")
            return context

        except Exception as e:
            tprint_warning(f"⚠️ Performance monitoring initialization failed: {e}")
            return None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _update_performance_metrics(self, result: AnalystPreMLResult, context: Optional[dict] = None):
        """Update performance metrics in the result."""
        if not UTILS_AVAILABLE or not self.config.enable_performance_monitoring:
            return

        try:
            current_memory = self._get_memory_usage()
            current_cpu = self._get_cpu_usage()

            result.memory_usage_peak_mb = max(result.memory_usage_peak_mb, current_memory)
            result.cpu_usage_peak_percent = max(result.cpu_usage_peak_percent, current_cpu)

            # Hardware optimization metrics
            result.hardware_optimization_enabled = self.config.enable_hardware_optimization
            result.m1_optimization_enabled = self.config.enable_m1_optimization and self.memory_optimizer is not None

            # Calculate parallel processing efficiency if applicable
            if self.config.enable_parallel_processing:
                result.parallel_processing_efficiency = safe_divide(
                    current_cpu, 100.0, default=0.0
                )  # Normalized efficiency

        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics update failed: {e}")

    async def orchestrate(
        self,
        training_data: pd.DataFrame,
        regime_assignments: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> AnalystPreMLResult:
        """
        Execute the complete pre-ML orchestration for Analyst models with enhanced utilities.

        Args:
            training_data: Input DataFrame with market data (60m timeframe)
            regime_assignments: Optional regime assignments for per-regime optimization
            **kwargs: Additional parameters

        Returns:
            AnalystPreMLResult with orchestrated features and metadata
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Enhanced Analyst Pre-ML Orchestration (60m timeframe)...")
        tprint_info(f"📊 Input data shape: {training_data.shape}")

        result = AnalystPreMLResult()
        result.total_samples = len(training_data)

        # Performance monitoring context
        monitoring_context = None

        try:
            # Validate input data
            await self._validate_input_data(training_data, regime_assignments)

            # Validate pre-training pipeline availability
            if not self.pre_training_pipeline:
                raise RuntimeError("Pre-training pipeline not available")

            # Initialize performance monitoring
            if self.config.enable_performance_monitoring:
                monitoring_context = await self._start_performance_monitoring()

            # Create sub-pipeline configuration with enhanced utilities
            sub_config = SubPipelineConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,  # 60m for Analyst
                data_dir=self.config.data_dir,
                parallel_processing=self.config.enable_parallel_processing,
                custom_params={
                    **self.config.custom_params,
                    'enable_per_regime_optimization': self.config.enable_per_regime_optimization,
                    'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                    'regime_assignments': regime_assignments,
                    'role': 'analyst',  # Mark as Analyst orchestration
                    'hardware_optimization_enabled': self.config.enable_hardware_optimization,
                    'data_quality_checker': self.data_quality_checker,
                    'tpe_optimizer': self.tpe_optimizer,
                    **kwargs
                }
            )

            tprint_info("📋 Enhanced Configuration:")
            tprint_info(f"  - Timeframe: {self.config.timeframe}")
            tprint_info(f"  - Hardware optimization: {self.config.enable_hardware_optimization}")
            tprint_info(f"  - M1 optimization: {self.config.enable_m1_optimization}")
            tprint_info(f"  - Bayesian TPE: {self.config.enable_bayesian_optimization}")
            tprint_info(f"  - Data quality checks: {self.config.enable_data_quality_checks}")
            tprint_info(f"  - Performance monitoring: {self.config.enable_performance_monitoring}")

            # Execute the complete pre-training pipeline using the new unified approach
            tprint_info("🚀 Executing complete pre-training pipeline...")
            result.phase = OrchestrationPhase.HORIZON_LABELING

            try:
                # Enable gate feature protection if available
                if GATE_PROTECTION_AVAILABLE and self.config.enable_gate_protection:
                    tprint_info("🛡️ Enabling gate feature protection for pre-training pipeline...")
                    enable_gate_protection()

                    # Add gate protection config to sub_config
                    if self.config.gate_protection_config:
                        sub_config.custom_params['gate_protection'] = self.config.gate_protection_config
                    else:
                        sub_config.custom_params['gate_protection'] = {
                            'enabled': True,
                            'max_gate_features_per_base': 3,
                            'min_gate_ic_improvement': 0.005,
                            'min_gate_stability': 0.4
                        }

                # Use Bayesian TPE optimization for feature selection if available
                if self.tpe_optimizer and self.config.enable_bayesian_optimization:
                    tprint_debug("🔬 Using Bayesian TPE optimization for feature selection")
                    # The TPE optimizer will be used within the feature selection pipeline
                    # through the custom_params passed to sub_config

                # Execute the complete pipeline
                pipeline_result = await self.pre_training_pipeline.execute_pipeline(sub_config)

                if not pipeline_result.get('success', False):
                    error_message = pipeline_result.get('error_message', 'Unknown pipeline error')
                    error_code = pipeline_result.get('error_code', 'unknown_error')
                    tprint_error(f"❌ Pre-training pipeline failed: [{error_code}] {error_message}")
                    self.logger.error(f"❌ Pre-training pipeline failed: [{error_code}] {error_message}")
                    raise RuntimeError(f"Pre-training pipeline failed ({error_code}): {error_message}")

                # Extract results from the pipeline execution
                results = pipeline_result.get('results', {})
                
                # Map results to our result structure
                result.horizon_labeling_result = results.get('analyst-labeler') or results.get('tactician-labeler')
                result.lookback_optimization_result = results.get('feature_generation_period_lookback_optimization_step')
                result.interactive_feature_generation_result = results.get('feature_generation_feature_generation_step')
                result.feature_selection_result = results.get('feature_generation_final_validation_step')

                # Extract final features and metadata
                final_features = results.get('feature_generation_final_validation_step', {}).get('artifacts', {})
                result.final_features = final_features.get('final_features')
                result.selected_feature_names = final_features.get('selected_features', [])

                if result.selected_feature_names:
                    result.final_feature_count = len(result.selected_feature_names)
                elif result.final_features is not None and hasattr(result.final_features, 'columns'):
                    result.final_feature_count = len(result.final_features.columns)
                else:
                    result.final_feature_count = 0

                # Set timing information (approximate breakdown)
                execution_time = pipeline_result.get('execution_time', 0.0)
                result.horizon_labeling_time = execution_time * 0.25
                result.lookback_optimization_time = execution_time * 0.25
                result.interactive_feature_generation_time = execution_time * 0.35
                result.feature_selection_time = execution_time * 0.15

                tprint_success(f"✅ Pre-training pipeline completed in {execution_time:.2f}s")
                tprint_success(f"✅ Final feature count: {result.final_feature_count}")

            except Exception as e:
                tprint_error(f"❌ Pre-training pipeline failed: {e}")
                raise

            # Mark as completed and update final metrics
            result.success = True
            result.phase = OrchestrationPhase.COMPLETED
            result.execution_time = tprint_timer(start_time)

            # Update performance metrics
            self._update_performance_metrics(result, monitoring_context)

            # Save result summary if requested
            if self.config.save_intermediate_results:
                self._save_result_summary(result)

            tprint_success(f"✅ Enhanced Analyst Pre-ML Orchestration completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Final feature count: {result.final_feature_count}")
            tprint_info(f"🧠 Memory peak: {result.memory_usage_peak_mb:.1f} MB")
            tprint_info(f"⚡ CPU peak: {result.cpu_usage_peak_percent:.1f}%")

            return result

        except Exception as e:
            result.success = False
            result.phase = OrchestrationPhase.FAILED
            result.error_message = str(e)
            result.execution_time = tprint_timer(start_time)

            # Update performance metrics even on failure
            self._update_performance_metrics(result, monitoring_context)

            tprint_error(f"❌ Enhanced Analyst Pre-ML Orchestration failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

        finally:
            # Clean up resources
            self._cleanup_hardware_optimization()

    def _save_result_summary(self, result: AnalystPreMLResult):
        """Save orchestration result summary for analysis."""
        try:
            summary_path = Path(self.config.output_directory) / "orchestration_summary.json"

            # Create summary dictionary
            summary = result.to_dict()
            summary['config_summary'] = {
                'timeframe': self.config.timeframe,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_bayesian_optimization': self.config.enable_bayesian_optimization,
                'enable_data_quality_checks': self.config.enable_data_quality_checks,
                'optimization_level': self.config.optimization_level
            }

            # Save with safe JSON operations
            safe_json_dump(summary, summary_path)
            tprint_debug(f"✅ Orchestration summary saved to {summary_path}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to save result summary: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the orchestrator."""
        return {
            'config': {
                'timeframe': self.config.timeframe,
                'enable_per_regime_optimization': self.config.enable_per_regime_optimization,
                'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_m1_optimization': self.config.enable_m1_optimization,
                'enable_bayesian_optimization': self.config.enable_bayesian_optimization,
                'optimization_level': self.config.optimization_level,
                'output_directory': self.config.output_directory
            },
            'component_availability': {
                'pre_training_pipeline': self.pre_training_pipeline is not None,
                'hardware_manager': self.hardware_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'data_quality_checker': self.data_quality_checker is not None,
                'tpe_optimizer': self.tpe_optimizer is not None,
                'utils_available': UTILS_AVAILABLE
            },
            'hardware_optimization': {
                'm1_available': self.config.enable_m1_optimization and self.memory_optimizer is not None,
                'gpu_acceleration': self.config.enable_gpu_acceleration,
                'optimization_level': self.config.optimization_level
            }
        }

    @contextmanager
    def orchestration_context(self):
        """
        Context manager for orchestration with automatic resource management.

        Usage:
            async with orchestrator.orchestration_context():
                result = await orchestrator.orchestrate(data)
        """
        try:
            tprint_debug("🔧 Entering orchestration context")
            yield self
        except Exception as e:
            tprint_error(f"❌ Orchestration context error: {e}")
            raise
        finally:
            tprint_debug("🧹 Cleaning up orchestration context")
            self._cleanup_hardware_optimization()

# Convenience function for external usage
async def execute_analyst_pre_ml_orchestration(
    training_data: pd.DataFrame,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[AnalystPreMLConfig] = None,
    **kwargs
) -> AnalystPreMLResult:
    """
    Execute Analyst pre-ML orchestration.

    Args:
        training_data: Input DataFrame with market data (60m timeframe)
        regime_assignments: Optional regime assignments for per-regime optimization
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        AnalystPreMLResult with orchestrated features and metadata
    """
    orchestrator = AnalystPreMLOrchestrator(config)
    return await orchestrator.orchestrate(training_data, regime_assignments, **kwargs)
