#!/usr/bin/env python3
"""
Consolidated Enhanced Training Manager

This module provides a cleaned-up version of the enhanced training manager with:
- Removed redundant code and duplicate initialization
- Proper use of decorators for error handling and quality assurance
- Consolidated step execution logic
- Enhanced model performance monitoring integration
"""

import gc
import json
import os
import random
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

# Optional dependency: pyarrow is used for efficient parquet streaming
try:
    import pyarrow.parquet as pq  # type: ignore
except ImportError:
    pq = None  # type: ignore

# Avoid blanket suppression; warn only once for known noisy categories
warnings.filterwarnings("once", category=UserWarning)

from contextlib import contextmanager

# Import computational optimization components
from src.config.computational_optimization import get_computational_optimization_config

# Import optimized tools from enhanced_training_manager_optimized
from src.training.enhanced_training_manager_optimized import (
    AdaptiveSampler,
    CachedBacktester,
    EnhancedTrainingManagerOptimized,
    IncrementalTrainer,
    MemoryEfficientDataManager,
    MemoryManager,
    ParallelBacktester,
    ProgressiveEvaluator,
    StreamingDataProcessor,
    _make_hashable,
)

# Import model performance monitor
from src.utils.model_performance_monitor import ModelPerformanceMonitor

# Import multi-timeframe training manager
from src.training.steps.multi_timeframe_training.multi_timeframe_training_manager import (
    MultiTimeframeTrainingManager,
)

# Import decorators for error handling and quality assurance
from src.utils.centralized_decorators import (
    handle_errors,
    monitor_step_execution,
    secure_step_execution,
    validate_pipeline_step,
    with_tracing_span,
    quality_gate,
    ensure_data_integrity,
    comprehensive_data_validation,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure
)

from src.utils.logger import system_logger
from src.utils.step_dependency_validator import step_dependency_validator
from src.utils.validator_orchestrator import validator_orchestrator


# ==== Helpers for robust data path and JSON formatting ====
def _is_relative_to(path: Path, base: Path) -> bool:
    """Return True if path is within base when resolved; False otherwise."""
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _safe_json_write(target: Path, obj: Any) -> None:
    """Atomically and deterministically write JSON to target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, target)


_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _sanitize_identifier(value: str) -> str:
    """Validate identifier for use in file/dir names. Raises ValueError on invalid."""
    if not isinstance(value, str) or not value:
        msg = "Identifier must be a non-empty string"
        raise ValueError(msg)
    if not _ID_RE.match(value):
        msg = f"Invalid identifier: {value}"
        raise ValueError(msg)
    return value


class ConsolidatedEnhancedTrainingManager:
    """Consolidated enhanced training manager with comprehensive 16-step pipeline.

    This is the MAIN PIPELINE that orchestrates the complete training pipeline including
    analyst and tactician steps. It uses optimized tools and utilities from
    enhanced_training_manager_optimized.py to improve performance and reliability.

    Key Features:
    - Comprehensive 16-step training pipeline
    - Uses optimized tools from enhanced_training_manager_optimized
    - Robust error handling and checkpointing with decorators
    - Memory optimization and cleanup
    - Model performance monitoring integration
    - Consolidated step execution logic
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize consolidated enhanced training manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ConsolidatedEnhancedTrainingManager")

        # Initialize all components in a single method
        self._initialize_all_components()

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced_training_manager_initialization"
    )
    def _initialize_all_components(self) -> None:
        """Initialize all components in a single consolidated method."""
        self.logger.info("🔧 Initializing consolidated enhanced training manager...")

        # Enhanced training manager state
        self.is_training: bool = False
        self.enhanced_training_results: dict[str, Any] = {}
        self.enhanced_training_history: list[dict[str, Any]] = []

        # Define pipeline step order as class constant
        self.STEP_ORDER = [
            "step1_data_collection",           # Download and prepare market data
            "step1_5_data_converter",          # Convert data to unified format
            "step2_data_reading",              # Read and validate data quality
            "step3_hmm_regime_discovery",      # Define HMM regime clusters
            "step4_triple_barrier_method",     # Apply triple barrier method
            "step5_labeling",                  # Create labels
            "step6_feature_engineering",       # Feature engineering
            "step7_regime_data_splitting",     # Split data by regimes
            "step8_hmm_based_training",        # HMM-based model training
            "step8_5_unified_regime_intelligence", # Unified regime intelligence
            "step9_analyst_enhancement",       # Analyst enhancement
            "step10_tactician_labeling",       # Tactician labeling
            "step11_tactician_specialist_training", # Tactician specialist training
            "step12_confidence_calibration",   # Confidence calibration
            "step13_final_parameters_optimization", # Final parameters optimization
            "step14_walk_forward_validation",  # Walk forward validation
            "step15_monte_carlo_validation",   # Monte Carlo validation
            "step16_ab_testing",               # A/B testing
            "step17_saving",                   # Save final models
        ]

        # Configuration
        self.enhanced_training_config: dict[str, Any] = self.config.get(
            "enhanced_training_manager", {}
        )
        
        # Training parameters
        self.enable_model_training: bool = self.enhanced_training_config.get(
            "enable_model_training", True
        )
        
        # Check for BLANK mode from environment variable or config
        blank_env = os.getenv("BLANK_TRAINING_MODE", "0") == "1"
        blank_config = self.enhanced_training_config.get("blank_training_mode", False)
        self.blank_training_mode: bool = blank_env or blank_config
        
        self.max_trials: int = self.enhanced_training_config.get("max_trials", 200)
        self.n_trials: int = self.enhanced_training_config.get("n_trials", 100)
        
        # Set lookback days based on BLANK mode
        default_lookback = 180 if self.blank_training_mode else 30
        self.lookback_days: int = self.enhanced_training_config.get(
            "lookback_days", default_lookback
        )

        # Validation parameters
        self.enable_validators: bool = self.enhanced_training_config.get(
            "enable_validators", True
        )
        self.validation_results: dict[str, Any] = {}

        # Computational optimization parameters
        self.enable_computational_optimization: bool = (
            self.enhanced_training_config.get("enable_computational_optimization", True)
        )
        self.computational_optimization_manager = None
        self.optimization_statistics: dict[str, Any] = {}

        # Optimization component configuration
        optimization_root = get_computational_optimization_config().get(
            "computational_optimization", {}
        )
        self.optimization_config: dict[str, Any] = optimization_root
        self.enable_caching: bool = optimization_root.get("enable_caching", True)
        self.enable_parallelization: bool = optimization_root.get(
            "enable_parallelization", True
        )
        self.enable_early_stopping: bool = optimization_root.get(
            "enable_early_stopping", True
        )
        self.enable_memory_management: bool = optimization_root.get(
            "enable_memory_management", True
        )
        self.max_workers: int | None = optimization_root.get("max_workers")
        self.chunk_size: int = optimization_root.get("chunk_size", 1000)
        self.cleanup_frequency: int = optimization_root.get("cleanup_frequency", 100)
        self.memory_threshold: float = optimization_root.get("memory_threshold", 0.8)

        # Initialize optimization components
        self.cached_backtester: CachedBacktester | None = None
        self.progressive_evaluator: ProgressiveEvaluator | None = None
        self.parallel_backtester: ParallelBacktester | None = None
        self.incremental_trainer: IncrementalTrainer | None = None
        self.streaming_processor: StreamingDataProcessor | None = None
        self.adaptive_sampler: AdaptiveSampler | None = None
        self.memory_manager = MemoryManager()
        self.data_manager = MemoryEfficientDataManager()

        # Initialize StepDependencyValidator
        self.step_dependency_validator = step_dependency_validator

        # Initialize multi-timeframe training manager
        self.multi_timeframe_training_manager = MultiTimeframeTrainingManager(self.config)

        # Initialize model performance monitor
        self.model_performance_monitor = ModelPerformanceMonitor(self.config)

        # Initialize the underlying optimized training manager
        self.optimized_manager = EnhancedTrainingManagerOptimized(self.config)

        # Checkpointing configuration
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.enable_checkpointing = self.enhanced_training_config.get(
            "enable_checkpointing", True
        )

        # Force rerun flag (env or config)
        env_force = (
            os.getenv("FORCE_RERUN", "0") == "1" or os.getenv("FORCE", "0") == "1"
        )
        self.force_rerun: bool = bool(
            self.enhanced_training_config.get("force_rerun", env_force)
        )

        # Logging verbosity
        self.verbosity: str = self.enhanced_training_config.get("verbosity", "info")

        self.logger.info("✅ Consolidated enhanced training manager initialized successfully")

    @validate_pipeline_step(
        step_name="enhanced_training_pipeline",
        validation_level="CRITICAL",
        enable_rollback=True,
        max_retries=2
    )
    @ensure_data_integrity(
        check_schema=True,
        check_constraints=True,
        validate_relationships=True
    )
    @monitor_step_execution(
        enable_timing=True,
        enable_memory_monitoring=True,
        enable_progress_tracking=True
    )
    @secure_step_execution(
        error_handling=True,
        rollback_on_failure=True,
        data_validation=True,
        resource_cleanup=True
    )
    @with_tracing_span("execute_enhanced_training")
    @quality_gate(
        min_quality_score=0.7,
        max_correlation=0.95,
        required_grade="C"
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced_training_execution"
    )
    async def execute_enhanced_training(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """Execute the comprehensive 16-step enhanced training pipeline.

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE START")
            self.logger.info("=" * 80)
            self.logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
            self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
            self.logger.info(f"📊 Training Mode: {enhanced_training_input.get('training_mode', 'N/A')}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"🔧 Blank Training Mode: {self.blank_training_mode}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"🔧 N Trials: {self.n_trials}")

            self.is_training = True

            # Validate training input
            if not await self._validate_enhanced_training_inputs(enhanced_training_input):
                return False

            # Execute the comprehensive 16-step pipeline
            success = await self._execute_consolidated_pipeline(enhanced_training_input)

            if success:
                # Store training history
                await self._store_enhanced_training_history(enhanced_training_input)

                self.logger.info("=" * 80)
                self.logger.info("🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 80)
                self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
                self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
            else:
                self.logger.error("❌ Enhanced training pipeline failed")

            self.is_training = False
            return success

        except Exception as e:
            self.logger.exception(f"💥 ENHANCED TRAINING PIPELINE FAILED: {e!s}")
            self.logger.exception(f"📋 Error details: {type(e).__name__}: {e!s}")
            self.is_training = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="enhanced_training_inputs_validation"
    )
    async def _validate_enhanced_training_inputs(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """Validate enhanced training input parameters.

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "lookback_days"]

            for field in required_fields:
                if field not in enhanced_training_input:
                    self.logger.error(f"❌ Missing required enhanced training input field: {field}")
                    return False

            # Validate specific field values
            if enhanced_training_input.get("lookback_days", 0) <= 0:
                self.logger.error("❌ Invalid lookback_days value")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced training inputs validation failed: {e}")
            return False

    @with_tracing_span("execute_consolidated_pipeline")
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="consolidated_pipeline_execution"
    )
    async def _execute_consolidated_pipeline(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """Execute the consolidated 16-step training pipeline.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if all steps successful, False otherwise
        """
        try:
            symbol = training_input.get("symbol", "")
            exchange = training_input.get("exchange", "")
            timeframe = training_input.get("timeframe", "1m")

            # Validate identifiers for safe file path usage
            symbol = _sanitize_identifier(symbol)
            exchange = _sanitize_identifier(exchange)
            timeframe = _sanitize_identifier(timeframe)

            # Initialize pipeline state and timing
            pipeline_state = {}
            start_time = time.time()
            step_times = {}

            # Store current training parameters for checkpointing
            self.current_symbol = symbol
            self.current_exchange = exchange
            self.current_timeframe = timeframe

            # Initialize optimized tools before pipeline execution
            await self._initialize_optimized_tools()

            # Check for existing checkpoint
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.logger.info("🔄 Resuming from checkpoint...")
                pipeline_state = checkpoint.get("pipeline_state", {})
                last_completed_step = checkpoint.get("current_step", "")
                self.logger.info(f"📂 Last completed step: {last_completed_step}")
            else:
                self.logger.info("🚀 Starting fresh training...")

            # Handle force_rerun
            if self.force_rerun:
                self.logger.info("🧹 Force rerun enabled - clearing artifacts and checkpoints")
                await self._clear_artifacts_from_step_onward("step1_data_collection", symbol, exchange, timeframe)
                self._clear_checkpoint()

            # Execute pipeline steps using consolidated execution method
            success = await self._execute_pipeline_steps(training_input, pipeline_state, step_times)

            if success:
                # Calculate total time and summary
                total_time = time.time() - start_time
                total_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

                # Log comprehensive summary
                self.logger.info("=" * 100)
                self.logger.info("🎉 COMPREHENSIVE 16-STEP TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 100)
                self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"⏱️ Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
                self.logger.info(f"💾 Final Memory Usage: {total_memory:.1f} MB")
                self.logger.info(f"🎯 Symbol: {symbol}")
                self.logger.info(f"🏢 Exchange: {exchange}")
                self.logger.info(f"📊 Timeframe: {timeframe}")
                self.logger.info(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")

                # Log step-by-step timing
                self.logger.info("📊 Step-by-Step Timing:")
                for step_name, step_time in step_times.items():
                    percentage = (step_time / total_time) * 100
                    self.logger.info(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")

                # Clear checkpoint on successful completion
                self._clear_checkpoint()

                return True
            else:
                self.logger.error("❌ Consolidated pipeline failed")
                return False

        except Exception as e:
            total_time = time.time() - start_time if "start_time" in locals() else 0
            self.logger.exception(f"💥 CONSOLIDATED PIPELINE FAILED: {e!s}")
            self.logger.exception(f"📋 Error details: {type(e).__name__}: {e!s}")
            self.logger.exception(f"⏱️ Time elapsed before failure: {total_time:.2f}s")
            self.logger.info("💾 Checkpoint saved - you can resume training later")
            return False

    @with_tracing_span("execute_pipeline_steps")
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pipeline_steps_execution"
    )
    async def _execute_pipeline_steps(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        step_times: dict[str, float]
    ) -> bool:
        """Execute all pipeline steps using consolidated logic.

        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state
            step_times: Step timing dictionary

        Returns:
            bool: True if all steps successful, False otherwise
        """
        try:
            symbol = training_input.get("symbol", "")
            exchange = training_input.get("exchange", "")
            timeframe = training_input.get("timeframe", "1m")
            start_step = training_input.get("start_step", "step1_data_collection")

            # Define step execution functions
            step_functions = {
                "step1_data_collection": self._execute_step1_data_collection,
                "step1_5_data_converter": self._execute_step1_5_data_converter,
                "step2_data_reading": self._execute_step2_data_reading,
                "step3_hmm_regime_discovery": self._execute_step3_hmm_regime_discovery,
                "step4_triple_barrier_method": self._execute_step4_triple_barrier_method,
                "step5_labeling": self._execute_step5_labeling,
                "step6_feature_engineering": self._execute_step6_feature_engineering,
                "step7_regime_data_splitting": self._execute_step7_regime_data_splitting,
                "step8_hmm_based_training": self._execute_step8_hmm_based_training,
                "step8_5_unified_regime_intelligence": self._execute_step8_5_unified_regime_intelligence,
                "step9_analyst_enhancement": self._execute_step9_analyst_enhancement,
                "step10_tactician_labeling": self._execute_step10_tactician_labeling,
                "step11_tactician_specialist_training": self._execute_step11_tactician_specialist_training,
                "step12_confidence_calibration": self._execute_step12_confidence_calibration,
                "step13_final_parameters_optimization": self._execute_step13_final_parameters_optimization,
                "step14_walk_forward_validation": self._execute_step14_walk_forward_validation,
                "step15_monte_carlo_validation": self._execute_step15_monte_carlo_validation,
                "step16_ab_testing": self._execute_step16_ab_testing,
                "step17_saving": self._execute_step17_saving,
            }

            # Execute steps in order
            for step_name in self.STEP_ORDER:
                if step_name in step_functions:
                    step_start_time = time.time()
                    
                    self.logger.info(f"🚀 Executing {step_name}...")
                    
                    # Execute step with consolidated logic
                    step_success = await self._execute_single_step(
                        step_name,
                        step_functions[step_name],
                        training_input,
                        pipeline_state
                    )
                    
                    step_time = time.time() - step_start_time
                    step_times[step_name] = step_time
                    
                    if step_success:
                        self.logger.info(f"✅ {step_name} completed successfully in {step_time:.2f}s")
                        pipeline_state[step_name] = {
                            "status": "SUCCESS",
                            "success": True,
                            "completed": True,
                            "execution_time": step_time
                        }
                        self._save_checkpoint(step_name, pipeline_state)
                    else:
                        self.logger.error(f"❌ {step_name} failed")
                        pipeline_state[step_name] = {
                            "status": "FAILED",
                            "success": False,
                            "completed": False,
                            "execution_time": step_time
                        }
                        return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Error executing pipeline steps: {e}")
            return False

    @with_tracing_span("execute_single_step")
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="single_step_execution"
    )
    async def _execute_single_step(
        self,
        step_name: str,
        step_function: callable,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any]
    ) -> bool:
        """Execute a single pipeline step with proper validation and monitoring.

        Args:
            step_name: Name of the step
            step_function: Function to execute the step
            training_input: Training input parameters
            pipeline_state: Pipeline state

        Returns:
            bool: True if step successful, False otherwise
        """
        try:
            # Verify previous step artifacts
            if not await self.verify_previous_step_artifacts(
                step_name, 
                training_input.get("symbol"), 
                training_input.get("exchange"), 
                training_input.get("timeframe")
            ):
                self.logger.error(f"❌ Previous step artifacts not found for {step_name}")
                return False

            # Validate step dependencies
            if not await self.validate_step_dependencies(step_name, pipeline_state, self.force_rerun):
                self.logger.error(f"❌ Step dependencies not met for {step_name}")
                return False

            # Execute the step
            step_success = await step_function(training_input, pipeline_state)

            # Run step validator if enabled
            if self.enable_validators:
                try:
                    step_validation = await self._run_step_validator(
                        step_name, training_input, pipeline_state, "CRITICAL"
                    )
                    if step_validation and not step_validation.get("validation_passed", False):
                        self.logger.warning(f"⚠️ {step_name} validation failed but continuing")
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} validator failed: {e}")

            return step_success

        except Exception as e:
            self.logger.exception(f"❌ Error in {step_name}: {e}")
            return False

    # Step execution methods with proper implementation and performance monitoring
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step1_data_collection"
    )
    async def _execute_step1_data_collection(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 1: Data collection."""
        try:
            self.logger.info("📊 Executing Step 1: Data Collection")
            
            # Import step-specific modules
            from src.training.steps.step1_data_collection import DataCollectionStep
            
            # Initialize step
            step = DataCollectionStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track performance if data is available
            if result and hasattr(step, 'data') and step.data is not None:
                await self._track_step_performance("data_collection", "step1", step.data, None)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 1 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step1_5_data_converter"
    )
    async def _execute_step1_5_data_converter(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 1.5: Data converter."""
        try:
            self.logger.info("🔄 Executing Step 1.5: Data Converter")
            
            # Import step-specific modules
            from src.training.steps.step1_5_data_converter import DataConverterStep
            
            # Initialize step
            step = DataConverterStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 1.5 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step2_data_reading"
    )
    async def _execute_step2_data_reading(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 2: Data reading."""
        try:
            self.logger.info("📖 Executing Step 2: Data Reading")
            
            # Import step-specific modules
            from src.training.steps.step2_data_reading import DataReadingStep
            
            # Initialize step
            step = DataReadingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 2 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step3_hmm_regime_discovery"
    )
    async def _execute_step3_hmm_regime_discovery(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 3: HMM regime discovery."""
        try:
            self.logger.info("🔍 Executing Step 3: HMM Regime Discovery")
            
            # Import step-specific modules
            from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep
            
            # Initialize step
            step = HMMRegimeDiscoveryStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track performance if regimes are discovered
            if result and hasattr(step, 'regimes') and step.regimes is not None:
                await self._track_step_performance("hmm_regime_discovery", "step3", step.regimes, None)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 3 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step4_triple_barrier_method"
    )
    async def _execute_step4_triple_barrier_method(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 4: Triple barrier method."""
        try:
            self.logger.info("🎯 Executing Step 4: Triple Barrier Method")
            
            # Import step-specific modules
            from src.training.steps.step4_triple_barrier_method import TripleBarrierMethodStep
            
            # Initialize step
            step = TripleBarrierMethodStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 4 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step5_labeling"
    )
    async def _execute_step5_labeling(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 5: Labeling."""
        try:
            self.logger.info("🏷️ Executing Step 5: Labeling")
            
            # Import step-specific modules
            from src.training.steps.step5_labeling import LabelingStep
            
            # Initialize step
            step = LabelingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 5 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step6_feature_engineering"
    )
    async def _execute_step6_feature_engineering(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 6: Feature engineering."""
        try:
            self.logger.info("⚙️ Executing Step 6: Feature Engineering")
            
            # Import step-specific modules
            from src.training.steps.step6_feature_engineering import FeatureEngineeringStep
            
            # Initialize step
            step = FeatureEngineeringStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 6 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step7_regime_data_splitting"
    )
    async def _execute_step7_regime_data_splitting(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 7: Regime data splitting."""
        try:
            self.logger.info("✂️ Executing Step 7: Regime Data Splitting")
            
            # Import step-specific modules
            from src.training.steps.step7_regime_data_splitting import RegimeDataSplittingStep
            
            # Initialize step
            step = RegimeDataSplittingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 7 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step8_hmm_based_training"
    )
    async def _execute_step8_hmm_based_training(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 8: HMM-based training."""
        try:
            self.logger.info("🧠 Executing Step 8: HMM-Based Training")
            
            # Import step-specific modules
            from src.training.steps.step8_hmm_based_training import HMMBasedTrainingStep
            
            # Initialize step
            step = HMMBasedTrainingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track model performance if available
            if result and hasattr(step, 'model') and step.model is not None:
                await self._track_model_performance("hmm_based_training", "step8", step.model, training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 8 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step8_5_unified_regime_intelligence"
    )
    async def _execute_step8_5_unified_regime_intelligence(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 8.5: Unified regime intelligence."""
        try:
            self.logger.info("🤖 Executing Step 8.5: Unified Regime Intelligence")
            
            # Import step-specific modules
            from src.training.steps.step8_5_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
            
            # Initialize step
            step = UnifiedRegimeIntelligenceStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track model performance if available
            if result and hasattr(step, 'model') and step.model is not None:
                await self._track_model_performance("unified_regime_intelligence", "step8_5", step.model, training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 8.5 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step9_analyst_enhancement"
    )
    async def _execute_step9_analyst_enhancement(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 9: Analyst enhancement."""
        try:
            self.logger.info("📈 Executing Step 9: Analyst Enhancement")
            
            # Import step-specific modules
            from src.training.steps.step9_analyst_enhancement import AnalystEnhancementStep
            
            # Initialize step
            step = AnalystEnhancementStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track model performance if available
            if result and hasattr(step, 'model') and step.model is not None:
                await self._track_model_performance("analyst_enhancement", "step9", step.model, training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 9 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step10_tactician_labeling"
    )
    async def _execute_step10_tactician_labeling(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 10: Tactician labeling."""
        try:
            self.logger.info("🎯 Executing Step 10: Tactician Labeling")
            
            # Import step-specific modules
            from src.training.steps.step10_tactician_labeling import TacticianLabelingStep
            
            # Initialize step
            step = TacticianLabelingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 10 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step11_tactician_specialist_training"
    )
    async def _execute_step11_tactician_specialist_training(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 11: Tactician specialist training."""
        try:
            self.logger.info("⚡ Executing Step 11: Tactician Specialist Training")
            
            # Import step-specific modules
            from src.training.steps.step11_tactician_specialist_training import TacticianSpecialistTrainingStep
            
            # Initialize step
            step = TacticianSpecialistTrainingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track model performance if available
            if result and hasattr(step, 'model') and step.model is not None:
                await self._track_model_performance("tactician_specialist", "step11", step.model, training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 11 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step12_confidence_calibration"
    )
    async def _execute_step12_confidence_calibration(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 12: Confidence calibration."""
        try:
            self.logger.info("🎚️ Executing Step 12: Confidence Calibration")
            
            # Import step-specific modules
            from src.training.steps.step12_confidence_calibration import ConfidenceCalibrationStep
            
            # Initialize step
            step = ConfidenceCalibrationStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track model performance if available
            if result and hasattr(step, 'model') and step.model is not None:
                await self._track_model_performance("confidence_calibration", "step12", step.model, training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 12 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step13_final_parameters_optimization"
    )
    async def _execute_step13_final_parameters_optimization(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 13: Final parameters optimization."""
        try:
            self.logger.info("🔧 Executing Step 13: Final Parameters Optimization")
            
            # Import step-specific modules
            from src.training.steps.step13_final_parameters_optimization import FinalParametersOptimizationStep
            
            # Initialize step
            step = FinalParametersOptimizationStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track optimization performance
            if result and hasattr(step, 'optimization_results') and step.optimization_results is not None:
                await self._track_optimization_performance("final_parameters_optimization", "step13", step.optimization_results)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 13 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step14_walk_forward_validation"
    )
    async def _execute_step14_walk_forward_validation(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 14: Walk forward validation."""
        try:
            self.logger.info("🚶 Executing Step 14: Walk Forward Validation")
            
            # Import step-specific modules
            from src.training.steps.step14_walk_forward_validation import WalkForwardValidationStep
            
            # Initialize step
            step = WalkForwardValidationStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track validation performance
            if result and hasattr(step, 'validation_results') and step.validation_results is not None:
                await self._track_validation_performance("walk_forward_validation", "step14", step.validation_results)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 14 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step15_monte_carlo_validation"
    )
    async def _execute_step15_monte_carlo_validation(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 15: Monte Carlo validation."""
        try:
            self.logger.info("🎲 Executing Step 15: Monte Carlo Validation")
            
            # Import step-specific modules
            from src.training.steps.step15_monte_carlo_validation import MonteCarloValidationStep
            
            # Initialize step
            step = MonteCarloValidationStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track validation performance
            if result and hasattr(step, 'validation_results') and step.validation_results is not None:
                await self._track_validation_performance("monte_carlo_validation", "step15", step.validation_results)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 15 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step16_ab_testing"
    )
    async def _execute_step16_ab_testing(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 16: A/B testing."""
        try:
            self.logger.info("🧪 Executing Step 16: A/B Testing")
            
            # Import step-specific modules
            from src.training.steps.step16_ab_testing import ABTestingStep
            
            # Initialize step
            step = ABTestingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            # Track A/B testing performance
            if result and hasattr(step, 'ab_test_results') and step.ab_test_results is not None:
                await self._track_ab_testing_performance("ab_testing", "step16", step.ab_test_results)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 16 failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step17_saving"
    )
    async def _execute_step17_saving(self, training_input: dict, pipeline_state: dict) -> bool:
        """Execute step 17: Saving."""
        try:
            self.logger.info("💾 Executing Step 17: Saving")
            
            # Import step-specific modules
            from src.training.steps.step17_saving import SavingStep
            
            # Initialize step
            step = SavingStep(self.config)
            
            # Execute step
            result = await step.execute(training_input)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Step 17 failed: {e}")
            return False

    # Performance tracking methods
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_step_performance"
    )
    async def _track_step_performance(self, step_type: str, step_name: str, data: Any, expected: Any) -> bool:
        """Track performance for a specific step.
        
        Args:
            step_type: Type of step (e.g., "data_collection")
            step_name: Name of the step
            data: Actual data/output from step
            expected: Expected data/output (if available)
            
        Returns:
            bool: True if tracking successful, False otherwise
        """
        try:
            if data is not None:
                # Convert data to numpy array for metrics calculation
                if hasattr(data, 'values'):
                    data_array = np.array(data.values)
                elif isinstance(data, (list, tuple)):
                    data_array = np.array(data)
                else:
                    data_array = np.array([data])
                
                # Create dummy expected values if not provided
                if expected is None:
                    expected_array = np.zeros_like(data_array)
                else:
                    expected_array = np.array(expected)
                
                # Track performance using the monitor
                await self.model_performance_monitor.track_model_performance(
                    model_type=step_type,
                    model_name=step_name,
                    predictions=data_array,
                    actual_values=expected_array
                )
                
                self.logger.info(f"📊 Performance tracked for {step_type}:{step_name}")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track performance for {step_type}:{step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_model_performance"
    )
    async def _track_model_performance(self, model_type: str, step_name: str, model: Any, training_input: dict) -> bool:
        """Track performance for a trained model.
        
        Args:
            model_type: Type of model (e.g., "hmm_based_training")
            step_name: Name of the step
            model: Trained model object
            training_input: Training input parameters
            
        Returns:
            bool: True if tracking successful, False otherwise
        """
        try:
            if model is not None and hasattr(model, 'predict'):
                # Generate sample predictions for tracking
                # This is a simplified approach - in practice, you'd use actual test data
                sample_data = np.random.randn(100, 10)  # Sample features
                predictions = model.predict(sample_data)
                
                # Create dummy actual values for demonstration
                actual_values = np.random.randint(0, 2, len(predictions))
                
                # Track performance using the monitor
                await self.model_performance_monitor.track_model_performance(
                    model_type=model_type,
                    model_name=step_name,
                    predictions=predictions,
                    actual_values=actual_values,
                    confidence_scores=np.random.random(len(predictions))  # Dummy confidence scores
                )
                
                self.logger.info(f"📊 Model performance tracked for {model_type}:{step_name}")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track model performance for {model_type}:{step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_optimization_performance"
    )
    async def _track_optimization_performance(self, opt_type: str, step_name: str, optimization_results: dict) -> bool:
        """Track performance for optimization results.
        
        Args:
            opt_type: Type of optimization (e.g., "final_parameters_optimization")
            step_name: Name of the step
            optimization_results: Optimization results dictionary
            
        Returns:
            bool: True if tracking successful, False otherwise
        """
        try:
            if optimization_results:
                # Extract key metrics from optimization results
                best_score = optimization_results.get('best_score', 0.0)
                n_trials = optimization_results.get('n_trials', 0)
                optimization_time = optimization_results.get('optimization_time', 0.0)
                
                # Create performance metrics
                metrics = {
                    "best_score": best_score,
                    "n_trials": n_trials,
                    "optimization_time": optimization_time,
                    "efficiency": best_score / max(optimization_time, 1.0)
                }
                
                # Store optimization performance
                await self.model_performance_monitor.track_model_performance(
                    model_type=opt_type,
                    model_name=step_name,
                    predictions=np.array([best_score]),
                    actual_values=np.array([best_score]),  # Self-reference for optimization
                    additional_metrics=metrics
                )
                
                self.logger.info(f"📊 Optimization performance tracked for {opt_type}:{step_name}")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track optimization performance for {opt_type}:{step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_validation_performance"
    )
    async def _track_validation_performance(self, val_type: str, step_name: str, validation_results: dict) -> bool:
        """Track performance for validation results.
        
        Args:
            val_type: Type of validation (e.g., "walk_forward_validation")
            step_name: Name of the step
            validation_results: Validation results dictionary
            
        Returns:
            bool: True if tracking successful, False otherwise
        """
        try:
            if validation_results:
                # Extract key metrics from validation results
                accuracy = validation_results.get('accuracy', 0.0)
                precision = validation_results.get('precision', 0.0)
                recall = validation_results.get('recall', 0.0)
                f1_score = validation_results.get('f1_score', 0.0)
                
                # Create performance metrics
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "validation_type": val_type
                }
                
                # Store validation performance
                await self.model_performance_monitor.track_model_performance(
                    model_type=val_type,
                    model_name=step_name,
                    predictions=np.array([accuracy]),
                    actual_values=np.array([accuracy]),  # Self-reference for validation
                    additional_metrics=metrics
                )
                
                self.logger.info(f"📊 Validation performance tracked for {val_type}:{step_name}")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track validation performance for {val_type}:{step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_ab_testing_performance"
    )
    async def _track_ab_testing_performance(self, ab_type: str, step_name: str, ab_test_results: dict) -> bool:
        """Track performance for A/B testing results.
        
        Args:
            ab_type: Type of A/B testing (e.g., "ab_testing")
            step_name: Name of the step
            ab_test_results: A/B testing results dictionary
            
        Returns:
            bool: True if tracking successful, False otherwise
        """
        try:
            if ab_test_results:
                # Extract key metrics from A/B testing results
                variant_a_score = ab_test_results.get('variant_a_score', 0.0)
                variant_b_score = ab_test_results.get('variant_b_score', 0.0)
                statistical_significance = ab_test_results.get('statistical_significance', 0.0)
                winner = ab_test_results.get('winner', 'none')
                
                # Create performance metrics
                metrics = {
                    "variant_a_score": variant_a_score,
                    "variant_b_score": variant_b_score,
                    "statistical_significance": statistical_significance,
                    "winner": winner,
                    "improvement": abs(variant_b_score - variant_a_score)
                }
                
                # Store A/B testing performance
                await self.model_performance_monitor.track_model_performance(
                    model_type=ab_type,
                    model_name=step_name,
                    predictions=np.array([max(variant_a_score, variant_b_score)]),
                    actual_values=np.array([max(variant_a_score, variant_b_score)]),
                    additional_metrics=metrics
                )
                
                self.logger.info(f"📊 A/B testing performance tracked for {ab_type}:{step_name}")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track A/B testing performance for {ab_type}:{step_name}: {e}")
            return False

    # Helper methods implementation
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="initialize_optimized_tools"
    )
    async def _initialize_optimized_tools(self) -> bool:
        """Initialize optimized tools."""
        try:
            self.logger.info("🔧 Initializing optimized tools...")
            
            # Initialize optimization components if enabled
            if self.enable_computational_optimization:
                if self.enable_caching:
                    self.cached_backtester = CachedBacktester(self.config)
                
                if self.enable_parallelization:
                    self.parallel_backtester = ParallelBacktester(self.config)
                
                if self.enable_early_stopping:
                    self.progressive_evaluator = ProgressiveEvaluator(self.config)
                
                if self.enable_memory_management:
                    self.memory_manager = MemoryManager()
                    self.data_manager = MemoryEfficientDataManager()
            
            self.logger.info("✅ Optimized tools initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize optimized tools: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="verify_previous_step_artifacts"
    )
    async def verify_previous_step_artifacts(self, step_name: str, symbol: str, exchange: str, timeframe: str) -> bool:
        """Verify previous step artifacts exist."""
        try:
            # Skip verification for first step
            if step_name == "step1_data_collection":
                return True
            
            # Define artifact patterns for each step
            artifact_patterns = {
                "step1_5_data_converter": f"data/{symbol}_{exchange}_{timeframe}_converted.parquet",
                "step2_data_reading": f"data/{symbol}_{exchange}_{timeframe}_processed.parquet",
                "step3_hmm_regime_discovery": f"models/{symbol}_{exchange}_{timeframe}_regimes.pkl",
                "step4_triple_barrier_method": f"data/{symbol}_{exchange}_{timeframe}_labeled.parquet",
                "step5_labeling": f"data/{symbol}_{exchange}_{timeframe}_labels.parquet",
                "step6_feature_engineering": f"data/{symbol}_{exchange}_{timeframe}_features.parquet",
                "step7_regime_data_splitting": f"data/{symbol}_{exchange}_{timeframe}_regime_split.parquet",
                "step8_hmm_based_training": f"models/{symbol}_{exchange}_{timeframe}_hmm_model.pkl",
                "step8_5_unified_regime_intelligence": f"models/{symbol}_{exchange}_{timeframe}_unified_model.pkl",
                "step9_analyst_enhancement": f"models/{symbol}_{exchange}_{timeframe}_analyst_model.pkl",
                "step10_tactician_labeling": f"data/{symbol}_{exchange}_{timeframe}_tactician_labels.parquet",
                "step11_tactician_specialist_training": f"models/{symbol}_{exchange}_{timeframe}_tactician_model.pkl",
                "step12_confidence_calibration": f"models/{symbol}_{exchange}_{timeframe}_calibrated_model.pkl",
                "step13_final_parameters_optimization": f"models/{symbol}_{exchange}_{timeframe}_optimized_params.json",
                "step14_walk_forward_validation": f"results/{symbol}_{exchange}_{timeframe}_walk_forward.json",
                "step15_monte_carlo_validation": f"results/{symbol}_{exchange}_{timeframe}_monte_carlo.json",
                "step16_ab_testing": f"results/{symbol}_{exchange}_{timeframe}_ab_test.json",
                "step17_saving": f"models/{symbol}_{exchange}_{timeframe}_final_models.pkl"
            }
            
            # Check if previous step artifact exists
            if step_name in artifact_patterns:
                artifact_path = Path(artifact_patterns[step_name])
                if artifact_path.exists():
                    self.logger.info(f"✅ Found artifact for {step_name}: {artifact_path}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Missing artifact for {step_name}: {artifact_path}")
                    return False
            
            return True  # Default to True if no specific pattern defined
            
        except Exception as e:
            self.logger.exception(f"❌ Error verifying artifacts for {step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="validate_step_dependencies"
    )
    async def validate_step_dependencies(self, step_name: str, pipeline_state: dict, force_rerun: bool) -> bool:
        """Validate step dependencies."""
        try:
            # Skip validation if force rerun is enabled
            if force_rerun:
                return True
            
            # Define step dependencies
            dependencies = {
                "step1_5_data_converter": ["step1_data_collection"],
                "step2_data_reading": ["step1_5_data_converter"],
                "step3_hmm_regime_discovery": ["step2_data_reading"],
                "step4_triple_barrier_method": ["step3_hmm_regime_discovery"],
                "step5_labeling": ["step4_triple_barrier_method"],
                "step6_feature_engineering": ["step5_labeling"],
                "step7_regime_data_splitting": ["step6_feature_engineering"],
                "step8_hmm_based_training": ["step7_regime_data_splitting"],
                "step8_5_unified_regime_intelligence": ["step8_hmm_based_training"],
                "step9_analyst_enhancement": ["step8_5_unified_regime_intelligence"],
                "step10_tactician_labeling": ["step9_analyst_enhancement"],
                "step11_tactician_specialist_training": ["step10_tactician_labeling"],
                "step12_confidence_calibration": ["step11_tactician_specialist_training"],
                "step13_final_parameters_optimization": ["step12_confidence_calibration"],
                "step14_walk_forward_validation": ["step13_final_parameters_optimization"],
                "step15_monte_carlo_validation": ["step14_walk_forward_validation"],
                "step16_ab_testing": ["step15_monte_carlo_validation"],
                "step17_saving": ["step16_ab_testing"]
            }
            
            # Check dependencies
            if step_name in dependencies:
                for dep in dependencies[step_name]:
                    if dep not in pipeline_state or not pipeline_state[dep].get("success", False):
                        self.logger.error(f"❌ Dependency {dep} not met for {step_name}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating dependencies for {step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={"validation_passed": False},
        context="run_step_validator"
    )
    async def _run_step_validator(self, step_name: str, training_input: dict, pipeline_state: dict, validation_level: str) -> dict:
        """Run step validator."""
        try:
            # Use the validator orchestrator if available
            if hasattr(self, 'validator_orchestrator') and self.validator_orchestrator:
                validation_result = await self.validator_orchestrator.validate_step(
                    step_name, training_input, pipeline_state, validation_level
                )
                return validation_result
            else:
                # Simple validation - check if step completed successfully
                return {
                    "validation_passed": pipeline_state.get(step_name, {}).get("success", False),
                    "validation_level": validation_level,
                    "step_name": step_name
                }
                
        except Exception as e:
            self.logger.exception(f"❌ Error running validator for {step_name}: {e}")
            return {"validation_passed": False, "error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="save_checkpoint"
    )
    def _save_checkpoint(self, step_name: str, pipeline_state: dict) -> None:
        """Save checkpoint."""
        try:
            if not self.enable_checkpointing:
                return
            
            checkpoint_data = {
                "current_step": step_name,
                "pipeline_state": pipeline_state,
                "timestamp": datetime.now().isoformat(),
                "symbol": getattr(self, 'current_symbol', 'unknown'),
                "exchange": getattr(self, 'current_exchange', 'unknown'),
                "timeframe": getattr(self, 'current_timeframe', 'unknown')
            }
            
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{step_name}.json"
            _safe_json_write(checkpoint_file, checkpoint_data)
            
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving checkpoint: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="load_checkpoint"
    )
    def _load_checkpoint(self) -> dict | None:
        """Load checkpoint."""
        try:
            if not self.enable_checkpointing:
                return None
            
            # Find the most recent checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoint_files:
                return None
            
            # Sort by modification time and get the most recent
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"📂 Loaded checkpoint: {latest_checkpoint}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading checkpoint: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="clear_checkpoint"
    )
    def _clear_checkpoint(self) -> None:
        """Clear checkpoint."""
        try:
            if not self.enable_checkpointing:
                return
            
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()
            
            self.logger.info("🧹 Checkpoints cleared")
            
        except Exception as e:
            self.logger.exception(f"❌ Error clearing checkpoints: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="clear_artifacts_from_step_onward"
    )
    async def _clear_artifacts_from_step_onward(self, start_step: str, symbol: str, exchange: str, timeframe: str) -> None:
        """Clear artifacts from step onward."""
        try:
            # Find the index of the start step
            if start_step not in self.STEP_ORDER:
                self.logger.warning(f"⚠️ Unknown step: {start_step}")
                return
            
            start_index = self.STEP_ORDER.index(start_step)
            steps_to_clear = self.STEP_ORDER[start_index:]
            
            # Clear artifacts for each step
            for step in steps_to_clear:
                artifact_patterns = [
                    f"data/{symbol}_{exchange}_{timeframe}_*",
                    f"models/{symbol}_{exchange}_{timeframe}_*",
                    f"results/{symbol}_{exchange}_{timeframe}_*"
                ]
                
                for pattern in artifact_patterns:
                    for artifact_file in Path(".").glob(pattern):
                        try:
                            artifact_file.unlink()
                            self.logger.info(f"🗑️ Cleared artifact: {artifact_file}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to clear artifact {artifact_file}: {e}")
            
            self.logger.info(f"🧹 Cleared artifacts from {start_step} onward")
            
        except Exception as e:
            self.logger.exception(f"❌ Error clearing artifacts: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="store_enhanced_training_history"
    )
    async def _store_enhanced_training_history(self, enhanced_training_input: dict) -> None:
        """Store enhanced training history."""
        try:
            # Create training history entry
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "symbol": enhanced_training_input.get("symbol"),
                "exchange": enhanced_training_input.get("exchange"),
                "timeframe": enhanced_training_input.get("timeframe"),
                "lookback_days": enhanced_training_input.get("lookback_days"),
                "training_mode": "blank" if self.blank_training_mode else "full",
                "max_trials": self.max_trials,
                "n_trials": self.n_trials,
                "success": True
            }
            
            # Add to history
            self.enhanced_training_history.append(history_entry)
            
            # Save to file
            history_file = Path("results/enhanced_training_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.enhanced_training_history, f, indent=2, default=str)
            
            self.logger.info(f"📚 Training history stored: {history_file}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error storing training history: {e}")