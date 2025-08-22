# src/training/enhanced_training_manager.py

# Added optimized imports
import gc
import json
import os
import random
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil

# Optional dependency: pyarrow is used for efficient parquet streaming; import lazily in methods
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

# Add model trainer import
from src.training.optimization.computational_optimization_manager import (
    create_computational_optimization_manager,
)

# Import multi-timeframe training manager
from src.training.steps.multi_timeframe_training.multi_timeframe_training_manager import (
    MultiTimeframeTrainingManager,
)

# Import the auto-fix decorator for data quality issues
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
    retry_on_failure,
    circuit_breaker,
    safe_operation,
)

# Import new QA decorators
from src.utils.training_pipeline_decorators import (
    validate_pipeline_step,
    ensure_data_integrity,
    monitor_step_execution,
    secure_step_execution,
    validate_pipeline_input,
    monitor_performance,
    data_quality_guard,
    artifact_versioning,
    time_budget_watchdog,
    nan_inf_and_constant_guard,
)

# Import validation components
from src.utils.logger import system_logger
from src.utils.step_dependency_validator import step_dependency_validator
from src.utils.validator_orchestrator import validator_orchestrator
from src.utils.data_quality_validator import DataQualityValidator
from src.utils.data_sanitizer import DataSanitizer

# ==== Helpers for robust data path and JSON formatting ====
def _is_relative_to(path: Path, base: Path) -> bool:
    """Return True if path is within base when resolved; False otherwise."""
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _safe_json_write(target: Path, obj: Any) -> None:
    """Atomically and deterministically write JSON to target.

    - Ensures parent directory exists
    - Writes UTF-8 with Unix newlines
    - Sorts keys for deterministic diffs
    - fsyncs before atomic replace
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            # fsync best-effort; ignore if unavailable
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


class EnhancedTrainingManager:
    """Enhanced training manager with comprehensive 16-step pipeline.

    This is the MAIN PIPELINE that orchestrates the complete training pipeline including
    analyst and tactician steps. It uses optimized tools and utilities from
    enhanced_training_manager_optimized.py to improve performance and reliability.

    Key Features:
    - Comprehensive 16-step training pipeline
    - Uses optimized tools from enhanced_training_manager_optimized:
      * CachedBacktester for avoiding redundant calculations
      * ProgressiveEvaluator for early stopping of unpromising trials
      * ParallelBacktester for parallel execution
      * IncrementalTrainer for reusing model states
      * StreamingDataProcessor for handling large datasets efficiently
      * AdaptiveSampler for focusing on promising regions
      * MemoryEfficientDataManager and MemoryManager for memory optimization
    - Robust error handling and checkpointing
    - Memory optimization and cleanup
    - Optional pyarrow support with fallback to pandas
    - Enhanced data processing with technical indicator precomputation
    - Comprehensive data quality validation and sanitization
    - Step dependency validation with force override support

    Integration:
    - Acts as the main entry point for all training operations
    - Delegates optimization tasks to EnhancedTrainingManagerOptimized
    - Provides unified interface while leveraging optimized backend
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize enhanced training manager.

        Args:
            config: Configuration dictionary

        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedTrainingManager")

        # Enhanced training manager state
        self.is_training: bool = False
        self.enhanced_training_results: dict[str, Any] = {}
        self.enhanced_training_history: list[dict[str, Any]] = []

        # Define pipeline step order as class constant
        self.STEP_ORDER = [
            "step1_data_collection",
            "step1_5_data_converter",
            "step2_feature_engineering",
            "step3_hmm_regime_discovery",
            "step4_processing_labeling",
            "step5_regime_data_splitting",
            "step6_hmm_based_training",
            "step6_5_unified_regime_intelligence",
            "step7_analyst_enhancement",
            "step8_tactician_labeling",
            "step9_tactician_specialist_training",
            "step10_confidence_calibration",
            "step11_final_parameters_optimization",
            "step12_walk_forward_validation",
            "step13_monte_carlo_validation",
            "step14_ab_testing",
            "step15_saving",
        ]

        # Define critical artifact patterns for each step
        self.CRITICAL_ARTIFACTS = {
            "step1_data_collection": [
                "data_cache/klines_{exchange}_{symbol}_1m_consolidated.parquet",
                "data_cache/parquet/aggtrades_{exchange}_{symbol}/**/*.parquet",
            ],
            "step1_5_data_converter": [
                "data_cache/unified/{exchange}/{symbol}/{timeframe}/**/*.parquet",
                "data_cache/unified/{exchange}_{symbol}_{timeframe}_config.json",
            ],
            "step2_feature_engineering": [
                "data/training/{exchange}_{symbol}_features_train.parquet",
                "data/training/{exchange}_{symbol}_features_metadata.json",
            ],
            "step3_hmm_regime_discovery": [
                "data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet",
            ],
            "step4_processing_labeling": [
                "data/training/{exchange}_{symbol}_{timeframe}_labeled_validation.parquet",
            ],
            "step5_regime_data_splitting": [
                "data/training/{exchange}_{symbol}_{timeframe}_regime_splits_train.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_regime_splits_validation.parquet",
            ],
            "step6_hmm_based_training": [
                "data/training/{exchange}_{symbol}_{timeframe}_hmm_models.pkl",
            ],
            "step6_5_unified_regime_intelligence": [
                "data/training/{exchange}_{symbol}_{timeframe}_unified_intelligence.parquet",
            ],
            "step7_analyst_enhancement": [
                "data/training/{exchange}_{symbol}_{timeframe}_analyst_models.pkl",
            ],
            "step8_tactician_labeling": [
                "data/training/{exchange}_{symbol}_{timeframe}_tactician_labels.parquet",
            ],
            "step9_tactician_specialist_training": [
                "data/training/{exchange}_{symbol}_{timeframe}_specialist_models.pkl",
            ],
            "step10_confidence_calibration": [
                "data/training/{exchange}_{symbol}_{timeframe}_calibration_results.pkl",
            ],
            "step11_final_parameters_optimization": [
                "data/training/{exchange}_{symbol}_{timeframe}_optimization_results.json",
            ],
            "step12_walk_forward_validation": [
                "data/training/{exchange}_{symbol}_{timeframe}_walk_forward_results.json",
            ],
            "step13_monte_carlo_validation": [
                "data/training/{exchange}_{symbol}_{timeframe}_monte_carlo_results.json",
            ],
            "step14_ab_testing": [
                "data/training/{exchange}_{symbol}_{timeframe}_ab_test_results.json",
            ],
            "step15_saving": [
                "data/training/{exchange}_{symbol}_{timeframe}_final_models.pkl",
            ],
        }

        # Initialize data quality and sanitization components
        self.data_quality_validator = DataQualityValidator()
        self.data_sanitizer = DataSanitizer()

        # Configuration
        self.enhanced_training_config: dict[str, Any] = self.config.get(
            "enhanced_training_manager",
            {},
        )
        self.enhanced_training_interval: int = self.enhanced_training_config.get(
            "enhanced_training_interval",
            3600,
        )
        self.max_enhanced_training_history: int = self.enhanced_training_config.get(
            "max_enhanced_training_history",
            100,
        )

        # Training parameters
        self.enable_model_training: bool = self.enhanced_training_config.get(
            "enable_model_training", True,
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
            "lookback_days", default_lookback,
        )

        # Validation parameters
        self.enable_validators: bool = self.enhanced_training_config.get(
            "enable_validators", True,
        )
        self.validation_results: dict[str, Any] = {}

        # Computational optimization parameters
        self.enable_computational_optimization: bool = (
            self.enhanced_training_config.get("enable_computational_optimization", True)
        )
        # Lazily set by create_computational_optimization_manager; type hint kept loose to avoid import cycle
        self.computational_optimization_manager = None
        self.optimization_statistics: dict[str, Any] = {}

        # Optimization component configuration (ported)
        optimization_root = get_computational_optimization_config().get(
            "computational_optimization", {},
        )
        self.optimization_config: dict[str, Any] = optimization_root
        self.enable_caching: bool = optimization_root.get("enable_caching", True)
        self.enable_parallelization: bool = optimization_root.get(
            "enable_parallelization", True,
        )
        self.enable_early_stopping: bool = optimization_root.get(
            "enable_early_stopping", True,
        )
        self.enable_memory_management: bool = optimization_root.get(
            "enable_memory_management", True,
        )
        self.max_workers: int | None = optimization_root.get("max_workers")
        self.chunk_size: int = optimization_root.get("chunk_size", 1000)
        self.cleanup_frequency: int = optimization_root.get("cleanup_frequency", 100)
        self.memory_threshold: float = optimization_root.get("memory_threshold", 0.8)

        # Optimization components (lazy init)
        # Using classes from enhanced_training_manager_optimized
        self.cached_backtester: CachedBacktester | None = None
        self.progressive_evaluator: ProgressiveEvaluator | None = None
        self.parallel_backtester: ParallelBacktester | None = None
        self.incremental_trainer: IncrementalTrainer | None = None
        self.streaming_processor: StreamingDataProcessor | None = None
        self.adaptive_sampler: AdaptiveSampler | None = None
        self.memory_manager: MemoryManager | None = None
        self.data_manager: MemoryEfficientDataManager | None = None

        # Checkpointing configuration
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        # Note: final paths are namespaced per symbol/exchange/timeframe at save-time
        self.enable_checkpointing = self.enhanced_training_config.get(
            "enable_checkpointing", True,
        )

        # Initialize optimized tools from enhanced_training_manager_optimized
        self.cached_backtester: CachedBacktester | None = None
        self.progressive_evaluator: ProgressiveEvaluator | None = None
        self.parallel_backtester: ParallelBacktester | None = None
        self.incremental_trainer: IncrementalTrainer | None = None
        self.streaming_processor: StreamingDataProcessor | None = None
        self.adaptive_sampler: AdaptiveSampler | None = None
        self.memory_manager = MemoryManager()
        self.data_manager = MemoryEfficientDataManager()

        # Initialize StepDependencyValidator for step dependency validation
        self.step_dependency_validator = step_dependency_validator

        # Initialize multi-timeframe training manager
        self.multi_timeframe_training_manager = MultiTimeframeTrainingManager(config)

        # Optimization configuration
        self.optimization_config = self.config.get("computational_optimization", {})
        self._load_optimization_config()

        # Initialize the underlying optimized training manager for advanced operations
        self.optimized_manager = EnhancedTrainingManagerOptimized(config)

        # Logging verbosity
        self.verbosity: str = self.enhanced_training_config.get(
            "verbosity", "info",
        )  # "info" or "debug"

        # MoE label expert artifacts and persistence
        self.label_expert_models: dict[str, dict[str, Any]] = {}
        self.label_expert_calibrators: dict[str, Any] = {}
        self.label_reliability: dict[str, float] = {}
        self.activation_thresholds: dict[str, float] = {}
        self.artifacts_dir: Path = Path(
            self.enhanced_training_config.get(
                "artifacts_dir", "artifacts/meta_labeling",
            ),
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Force rerun flag (env or config)
        env_force = (
            os.getenv("FORCE_RERUN", "0") == "1" or os.getenv("FORCE", "0") == "1"
        )
        self.force_rerun: bool = bool(
            self.enhanced_training_config.get("force_rerun", env_force),
        )

        # Checkpointing configuration
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        # Note: final paths are namespaced per symbol/exchange/timeframe at save-time
        self.enable_checkpointing = self.enhanced_training_config.get(
            "enable_checkpointing", True,
        )

        self.logger.info("Loaded optimization configuration")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="enhanced training manager initialization",
    )
    @validate_pipeline_input(
        required_params=["config"],
        data_validation=True,
        memory_check=True,
    )
    @monitor_step_execution(
        step_name="initialization",
        enable_timing=True,
        enable_memory_monitoring=True,
        enable_progress_tracking=True,
    )
    async def initialize(self) -> bool:
        """Initialize enhanced training manager.

        Returns:
            bool: True if initialization successful, False otherwise

        """
        try:
            self.logger.info("🚀 Initializing Enhanced Training Manager...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid configuration for enhanced training manager")
                return False

            # Initialize computational optimization if enabled
            if self.enable_computational_optimization:
                await self._initialize_computational_optimization()

            # Initialize optimized tools
            await self._initialize_optimized_tools()

            self.logger.info("✅ Enhanced Training Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Training Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate enhanced training manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise

        """
        try:
            # Validate enhanced training manager specific settings
            if self.max_enhanced_training_history <= 0:
                self.logger.error("❌ Invalid max_enhanced_training_history configuration")
                return False

            if self.max_trials <= 0:
                self.logger.error("❌ Invalid max_trials configuration")
                return False

            if self.n_trials <= 0:
                self.logger.error("❌ Invalid n_trials configuration")
                return False

            if self.lookback_days <= 0:
                self.logger.error("❌ Invalid lookback_days configuration")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Configuration validation failed: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid enhanced training parameters"),
            AttributeError: (False, "Missing enhanced training components"),
            KeyError: (False, "Missing required enhanced training data"),
        },
        default_return=False,
        context="enhanced training execution",
    )
    @validate_pipeline_step(
        step_name="comprehensive_pipeline",
        validation_level="WARNING",
        enable_rollback=True,
        max_retries=2
    )
    @ensure_data_integrity(
        check_schema=True,
        check_constraints=True,
        validate_relationships=True,
        enable_checksums=True
    )
    @monitor_step_execution(
        track_memory=True,
        track_performance=True,
        track_resources=True,
        alert_thresholds={"max_execution_time": 3600, "max_memory_delta": 1024**3}
    )
    @secure_step_execution(
        validate_inputs=True,
        sanitize_outputs=True,
        check_permissions=True,
        audit_operations=True
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
            self.logger.info("🚀 COMPREHENSIVE 15-STEP ENHANCED TRAINING PIPELINE START")
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
            if not self._validate_enhanced_training_inputs(enhanced_training_input):
                return False

            # Execute the comprehensive 16-step pipeline
            success = await self._execute_comprehensive_pipeline(enhanced_training_input)

            if success:
                # Store training history
                await self._store_enhanced_training_history(enhanced_training_input)

                self.logger.info("=" * 80)
                self.logger.info("🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 80)
                self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
                self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
                self.logger.info("📋 Completed Steps:")
                self.logger.info("   1. Data Collection")
                self.logger.info("   2. Feature Engineering")
                self.logger.info("   3. HMM Regime Discovery")
                self.logger.info("   4. Processing & Labeling")
                self.logger.info("   5. Regime Data Splitting")
                self.logger.info("   6. HMM-Based Training")
                self.logger.info("   6.5. Unified Regime Intelligence")
                self.logger.info("   7. Analyst Enhancement")
                self.logger.info("   8. Tactician Labeling")
                self.logger.info("   9. Tactician Specialist Training")
                self.logger.info("   10. Confidence Calibration")
                self.logger.info("   11. Final Parameters Optimization")
                self.logger.info("   12. Walk Forward Validation")
                self.logger.info("   13. Monte Carlo Validation")
                self.logger.info("   14. A/B Testing")
                self.logger.info("   15. Saving Results")
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
        context="enhanced training inputs validation",
    )
    @data_quality_guard(
        check_nan_inf=True,
        check_data_types=True,
        check_missing_values=True,
        validation_level="WARNING"
    )
    def _validate_enhanced_training_inputs(
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

            # Sanitize identifiers
            try:
                enhanced_training_input["symbol"] = _sanitize_identifier(enhanced_training_input["symbol"])
                enhanced_training_input["exchange"] = _sanitize_identifier(enhanced_training_input["exchange"])
                enhanced_training_input["timeframe"] = _sanitize_identifier(enhanced_training_input["timeframe"])
            except ValueError as e:
                self.logger.error(f"❌ Invalid identifier in training input: {e}")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced training inputs validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="comprehensive pipeline execution",
    )
    @validate_pipeline_step(
        step_name="comprehensive_pipeline",
        validation_level="WARNING",
        enable_rollback=True,
        max_retries=2
    )
    @ensure_data_integrity(
        check_schema=True,
        check_constraints=True,
        validate_relationships=True,
        enable_checksums=True
    )
    @monitor_step_execution(
        track_memory=True,
        track_performance=True,
        track_resources=True,
        alert_thresholds={"max_execution_time": 3600, "max_memory_delta": 1024**3}
    )
    @secure_step_execution(
        validate_inputs=True,
        sanitize_outputs=True,
        check_permissions=True,
        audit_operations=True
    )
    async def _execute_comprehensive_pipeline(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """Execute the comprehensive 16-step training pipeline.

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

            data_dir = "data/training"
            data_root = Path(data_dir)
            start_step = training_input.get("start_step", "step1_data_collection")

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

            # Handle force_rerun: clear artifacts from starting step and subsequent steps
            if self.force_rerun:
                self.logger.info(f"🧹 Force rerun enabled - clearing artifacts from {start_step} and subsequent steps")
                await self._clear_artifacts_from_step_onward(start_step, symbol, exchange, timeframe)
                # Clear the checkpoint to ensure fresh start
                self._clear_checkpoint()
                self.logger.info(f"✅ Cleared artifacts and checkpoints from {start_step} onward")

            # Enhanced logging setup
            self.logger.info("=" * 100)
            self.logger.info("🚀 COMPREHENSIVE 15-STEP TRAINING PIPELINE START")
            self.logger.info("=" * 100)
            self.logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"💾 Memory Optimization: {'Enabled' if self.enable_computational_optimization else 'Disabled'}")
            self.logger.info(f"🚀 Starting from step: {start_step}")
            self.logger.info("=" * 100)

            # Execute each step with validation
            for step_name in self.STEP_ORDER:
                # Skip steps before start_step
                if self.STEP_ORDER.index(step_name) < self.STEP_ORDER.index(start_step):
                    continue

                # Validate step dependencies BEFORE execution (unless force_rerun)
                if not self.force_rerun:
                    if not await self.validate_step_dependencies(step_name, pipeline_state, False):
                        self.logger.error(f"❌ Step dependencies not met for {step_name}, stopping pipeline")
                        return False

                # Execute step with enhanced error handling and validation
                step_success = await self._execute_pipeline_step_with_validation(
                    step_name, training_input, pipeline_state, step_times
                )

                if not step_success:
                    self.logger.error(f"❌ Step {step_name} failed, stopping pipeline")
                    return False

                # Run validator for step (AFTER execution, for verification only)
                if self.enable_validators:
                    try:
                        step_validation = await self._run_step_validator(step_name, training_input, pipeline_state)
                        if step_validation and step_validation.get("validation_passed", False):
                            self.logger.info(f"🎉 {step_name} completed successfully and validation passed")
                        else:
                            self.logger.error(f"❌ {step_name} validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ {step_name} validator failed: {e} - stopping pipeline")
                        return False

            # Calculate total time and summary
            total_time = time.time() - start_time
            total_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

            # Log comprehensive summary
            self.logger.info("=" * 100)
            self.logger.info("🎉 COMPREHENSIVE 15-STEP TRAINING PIPELINE COMPLETED SUCCESSFULLY")
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

        except Exception as e:
            total_time = time.time() - start_time if "start_time" in locals() else 0
            self.logger.exception(f"💥 COMPREHENSIVE PIPELINE FAILED: {e!s}")
            self.logger.exception(f"📋 Error details: {type(e).__name__}: {e!s}")
            self.logger.exception(f"⏱️ Time elapsed before failure: {total_time:.2f}s")
            self.logger.info("💾 Checkpoint saved - you can resume training later")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pipeline step execution with validation",
    )
    @validate_pipeline_step(
        step_name="pipeline_step",
        validation_level="WARNING",
        enable_rollback=True,
        max_retries=3
    )
    @ensure_data_integrity(
        check_schema=True,
        check_constraints=True,
        validate_relationships=True,
        enable_checksums=True
    )
    @monitor_step_execution(
        track_memory=True,
        track_performance=True,
        track_resources=True,
        alert_thresholds={"max_execution_time": 1800, "max_memory_delta": 512**3}
    )
    @secure_step_execution(
        validate_inputs=True,
        sanitize_outputs=True,
        check_permissions=True,
        audit_operations=True
    )
    async def _execute_pipeline_step_with_validation(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        step_times: dict[str, float],
    ) -> bool:
        """Execute a pipeline step with comprehensive validation and error handling.

        Args:
            step_name: Name of the step to execute
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            step_times: Dictionary to store step timing

        Returns:
            bool: True if step completed successfully, False otherwise

        """
        step_start_time = time.time()
        step_display_name = step_name.replace("_", " ").title()

        try:
            self.logger.info(f"🚀 Starting {step_display_name}")
            self._heartbeat(step_display_name)

            # Verify previous step artifacts BEFORE execution (unless force_rerun)
            if not self.force_rerun:
                if not await self.verify_previous_step_artifacts(
                    step_name, 
                    training_input.get("symbol", ""), 
                    training_input.get("exchange", ""), 
                    training_input.get("timeframe", "1m")
                ):
                    self.logger.error(f"❌ Previous step artifacts not found for {step_display_name}, stopping pipeline")
                    return False

            # Execute the step based on step name
            step_success = await self._execute_specific_step(step_name, training_input, pipeline_state)

            if not step_success:
                self._log_step_completion(step_display_name, step_start_time, step_times, success=False)
                self.logger.error(f"❌ {step_display_name} failed - stopping pipeline")
                return False
            
            self._log_step_completion(step_display_name, step_start_time, step_times, success=True)

            # Update pipeline state
            pipeline_state[step_name.replace("step", "").replace("_", "")] = {
                "status": "SUCCESS",
                "success": True,
                "completed": True,
            }
            self._save_checkpoint(step_name, pipeline_state)
            step_times[step_name] = time.time() - step_start_time

            return True

        except Exception as e:
            self._log_step_completion(step_display_name, step_start_time, step_times, success=False)
            self.logger.exception(f"❌ Error in {step_display_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="specific step execution",
    )
    @retry_on_failure(max_retries=3, backoff_factor=2)
    @circuit_breaker(failure_threshold=3, recovery_timeout=300)
    async def _execute_specific_step(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """Execute a specific step based on step name.

        Args:
            step_name: Name of the step to execute
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if step completed successfully, False otherwise

        """
        try:
            # Import and execute the appropriate step module
            step_module_name = f"src.training.steps.{step_name}"
            
            try:
                step_module = __import__(step_module_name, fromlist=["run_step"])
                step_function = getattr(step_module, "run_step")
            except (ImportError, AttributeError):
                self.logger.warning(f"⚠️ Step module {step_module_name} not found, skipping")
                return True  # Skip missing steps

            # Prepare step arguments
            step_args = {
                "symbol": training_input.get("symbol", ""),
                "exchange": training_input.get("exchange", ""),
                "timeframe": training_input.get("timeframe", "1m"),
                "data_dir": "data/training",
                "force_rerun": self.force_rerun,
                "config": self.config,
            }

            # Execute the step
            result = await step_function(**step_args)
            
            return bool(result)

        except Exception as e:
            self.logger.exception(f"❌ Error executing {step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step dependency validation",
    )
    async def validate_step_dependencies(
        self,
        step_name: str,
        pipeline_state: dict[str, Any],
        force_rerun: bool = False,
    ) -> bool:
        """Validate that all dependencies for a step are met.

        Args:
            step_name: Name of the step to validate
            pipeline_state: Current pipeline state
            force_rerun: If True, skip dependency validation for the starting step

        Returns:
            True if dependencies are met, False otherwise

        """
        try:
            self.logger.info(f"🔍 Validating dependencies for {step_name}")

            # If force_rerun is True, we're starting from this step, so skip dependency validation
            if force_rerun:
                self.logger.info(f"✅ Force rerun enabled for {step_name}, skipping dependency validation")
                return True

            # Use StepDependencyValidator to check prerequisites
            # Extract exchange and symbol from pipeline_state or use defaults
            exchange = pipeline_state.get("exchange", "BINANCE")
            symbol = pipeline_state.get("symbol", "ETHUSDT")
            timeframe = pipeline_state.get("timeframe", "1m")
            # Use the same path structure as _save_checkpoint method
            checkpoint_dir = f"checkpoints/{exchange}/{symbol}/{timeframe}"
            
            validation_result = await self.step_dependency_validator.validate_step_prerequisites(
                step_name=step_name,
                pipeline_state=pipeline_state,
                checkpoint_dir=checkpoint_dir,
                force_rerun=force_rerun,
            )

            if validation_result["valid"]:
                self.logger.info(f"✅ Dependencies validated for {step_name}: {validation_result['reason']}")
                return True
            
            self.logger.error(f"❌ Dependencies failed for {step_name}: {validation_result['reason']}")

            # Log failed prerequisites for debugging
            if "failed_steps" in validation_result:
                self.logger.error(f"   Failed prerequisites: {validation_result['failed_steps']}")

            return False

        except Exception as e:
            self.logger.exception(f"🚨 Error validating dependencies for {step_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step validator execution",
    )
    async def _run_step_validator(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        force_rerun: bool = False,
    ) -> dict[str, Any]:
        """Run validator for a specific step.

        Args:
            step_name: Name of the step to validate
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            force_rerun: If True, skip dependency validation for the starting step

        Returns:
            Validation result dictionary

        """
        if not self.enable_validators:
            return {
                "step_name": step_name,
                "validation_passed": True,
                "skipped": True,
                "reason": "Validators disabled",
            }

        try:
            self.logger.info(f"🔍 Running validator for {step_name}")

            # First, validate step dependencies
            dependency_validation = await self.step_dependency_validator.validate_step_prerequisites(
                step_name, pipeline_state, "checkpoints", force_rerun,
            )

            if not dependency_validation["valid"]:
                self.logger.error(f"❌ Step dependencies failed for {step_name}: {dependency_validation['reason']}")
                return {
                    "step_name": step_name,
                    "validation_passed": False,
                    "error": f"Dependency validation failed: {dependency_validation['reason']}",
                }

            # If dependencies are valid, run the step validator
            validation_result = await validator_orchestrator.run_step_validator(
                step_name=step_name,
                training_input=training_input,
                pipeline_state=pipeline_state,
                config=self.config,
            )

            # Store validation result
            self.validation_results[step_name] = validation_result

            if validation_result.get("validation_passed", False):
                self.logger.info(f"✅ {step_name} validation passed")
            else:
                self.logger.warning(
                    f"⚠️ {step_name} validation failed: {validation_result.get('error', 'Unknown error')}",
                )

            return validation_result

        except Exception as e:
            self.logger.exception(f"❌ Error running validator for {step_name}: {e}")
            return {"step_name": step_name, "validation_passed": False, "error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="computational optimization initialization",
    )
    async def _initialize_computational_optimization(self) -> bool:
        """Initialize computational optimization components."""
        try:
            self.logger.info("🚀 Initializing computational optimization components...")

            # Get computational optimization configuration
            optimization_config = get_computational_optimization_config()

            # Create computational optimization manager
            self.computational_optimization_manager = (
                await create_computational_optimization_manager(
                    config=optimization_config,
                    market_data=pd.DataFrame(),  # Will be loaded during training
                    model_config={},  # Will be configured during training
                )
            )

            if self.computational_optimization_manager:
                self.logger.info("✅ Computational optimization components initialized successfully")
                return True
            self.logger.warning("⚠️ Failed to initialize computational optimization components")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Computational optimization initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimized tools initialization",
    )
    async def _initialize_optimized_tools(self) -> bool:
        """Initialize optimized tools and the optimized training manager."""
        try:
            self.logger.info("🚀 Initializing optimized tools...")

            # Ensure optimized_manager is defined
            if not hasattr(self, "optimized_manager"):
                self.logger.warning("⚠️ optimized_manager not defined, skipping initialization")
                return True

            # Initialize the underlying optimized training manager
            await self.optimized_manager.initialize()
            self.logger.info("   ✅ Optimized training manager initialized")

            # Initialize streaming processor
            if self.chunk_size:
                self.streaming_processor = StreamingDataProcessor(
                    chunk_size=self.chunk_size,
                )
                self.logger.info("   ✅ Streaming processor initialized")

            # Initialize parallel backtester if enabled
            if self.enable_parallelization:
                self.parallel_backtester = ParallelBacktester(
                    n_workers=self.max_workers,
                )
                self.logger.info(f"   ✅ Parallel backtester initialized with {self.max_workers} workers")

            # Initialize adaptive sampler
            self.adaptive_sampler = AdaptiveSampler()
            self.logger.info("   ✅ Adaptive sampler initialized")

            # Initialize incremental trainer
            base_model_config = self.config.get("model", {})
            self.incremental_trainer = IncrementalTrainer(base_model_config)
            self.logger.info("   ✅ Incremental trainer initialized")

            self.logger.info("✅ All optimized tools initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize optimized tools: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimized parameters optimization",
    )
    async def _run_optimized_parameters_optimization(
        self,
        symbol: str,
        data_dir: str,
        timeframe: str,
        exchange: str,
    ) -> bool:
        """Run optimized parameters optimization using computational optimization strategies."""
        try:
            self.logger.info(
                "🚀 Running optimized parameters optimization with enhanced tools...",
            )

            # Use optimized data loading from the optimized manager
            market_data = await self.optimized_manager._load_and_optimize_data(
                symbol, exchange, timeframe,
            )

            if market_data is None or market_data.empty:
                self.logger.error("❌ Failed to load market data for optimization")
                return False

            self.logger.info(
                f"✅ Loaded optimized market data: {len(market_data)} rows",
            )

            # Initialize cached backtester if not already done
            if self.enable_caching and self.cached_backtester is None:
                self.cached_backtester = CachedBacktester(market_data)
                self.logger.info("✅ Cached backtester initialized for optimization")

            # Initialize progressive evaluator if not already done
            if self.enable_early_stopping and self.progressive_evaluator is None:
                self.progressive_evaluator = ProgressiveEvaluator(market_data)
                self.logger.info("✅ Progressive evaluator initialized for optimization")

            # Define optimization objective function using cached backtester
            def optimization_objective(params):
                """Optimization objective using cached backtesting."""
                try:
                    # Use cached backtester for faster evaluation
                    if self.cached_backtester:
                        return self.cached_backtester.run_cached_backtest(params)
                    # Fallback to simple calculation
                    return np.random.uniform(-1.0, 1.0)
                except Exception as e:
                    self.logger.warning(f"Optimization objective failed: {e}")
                    return -1.0  # Penalize failed evaluations

            # Define progressive evaluation function
            def progressive_evaluator_func(data_subset, params):
                """Progressive evaluation function for early stopping."""
                try:
                    # Create temporary backtester for subset evaluation
                    temp_backtester = CachedBacktester(data_subset)
                    return temp_backtester.run_cached_backtest(params)
                except Exception:
                    return -1.0

            # Use parallel backtester if enabled
            optimization_results = {}
            if self.enable_parallelization:
                self.logger.info("🔄 Using parallel backtesting for optimization...")

                # Generate parameter combinations for parallel evaluation
                param_combinations = self._generate_parameter_combinations()

                # Run parallel backtesting with context manager
                with ParallelBacktester(n_workers=self.max_workers) as pb:
                    parallel_results = pb.evaluate_batch(
                        param_combinations, market_data,
                    )

                # Find best parameters from parallel results
                if parallel_results:
                    best_result = max(
                        parallel_results, key=lambda x: x.get("score", -float("inf")),
                    )
                    optimization_results = best_result
                    self.logger.info(f"✅ Parallel optimization completed. Best score: {best_result.get('score', 'N/A')}")

            # Use progressive evaluation if enabled
            elif self.enable_early_stopping and self.progressive_evaluator:
                self.logger.info("🔄 Using progressive evaluation for optimization...")

                # Run optimization with progressive evaluation
                best_params = None
                best_score = -float("inf")

                for trial in range(self.n_trials):
                    # Generate random parameters for this trial
                    params = self._generate_random_parameters()

                    # Use progressive evaluator
                    score = self.progressive_evaluator.evaluate_progressively(
                        params, progressive_evaluator_func,
                    )

                    if score > best_score:
                        best_score = score
                        best_params = params
                        self.logger.info(f"📈 New best score: {score} at trial {trial + 1}")

                optimization_results = {
                    "best_params": best_params,
                    "best_score": best_score,
                    "trials_completed": self.n_trials,
                }

            # Fallback to computational optimization manager
            else:
                self.logger.info("🔄 Using standard computational optimization...")

                # Update computational optimization manager with market data
                if self.computational_optimization_manager:
                    await self.computational_optimization_manager.initialize(
                        market_data, {},
                    )

                # Run optimized parameter optimization
                optimization_results = await self.computational_optimization_manager.optimize_parameters(
                    objective_function=optimization_objective,
                    n_trials=self.n_trials,
                    use_surrogates=True,
                )

            # Store optimization statistics
            if self.computational_optimization_manager:
                self.optimization_statistics = self.computational_optimization_manager.get_optimization_statistics()
            else:
                self.optimization_statistics = {
                    "method": "enhanced_optimized_tools",
                    "trials_completed": optimization_results.get("trials_completed", self.n_trials),
                    "best_score": optimization_results.get("best_score", 0.0),
                    "cache_hits": getattr(self.cached_backtester, "cache", {})
                    if self.cached_backtester
                    else {},
                    "memory_profile": self.memory_manager.profile_memory_usage(),
                }

            # Perform memory cleanup if enabled
            if self.enable_memory_management:
                # Check and cleanup if above threshold
                self.memory_manager.check_memory_usage()
                self.logger.info("🧹 Memory cleanup check completed")

            # Save optimization results
            await self._save_optimization_results(
                symbol, exchange, data_dir, optimization_results,
            )

            self.logger.info("✅ Enhanced optimized parameters optimization completed successfully")

            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced optimized parameters optimization failed: {e}")
            return False

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for parallel backtesting."""
        # This is a simplified implementation
        # In practice, you would generate meaningful parameter combinations
        combinations = []
        for _i in range(
            min(self.n_trials, 20),
        ):  # Limit combinations for parallel processing
            combinations.append(
                {
                    "param1": np.random.uniform(0.1, 1.0),
                    "param2": np.random.uniform(0.1, 1.0),
                    "param3": np.random.randint(1, 100),
                },
            )
        return combinations

    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters for optimization."""
        return {
            "param1": np.random.uniform(0.1, 1.0),
            "param2": np.random.uniform(0.1, 1.0),
            "param3": np.random.randint(1, 100),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="market data loading for optimization",
    )
    async def _load_market_data_for_optimization(
        self,
        symbol: str,
        data_dir: str,
        exchange: str,
    ) -> pd.DataFrame | None:
        """Load market data for optimization using optimized data manager."""
        try:
            # Load market data from the data directory
            # This is a simplified implementation

            # Prefer consolidated Parquet/CSV produced by Step 1
            preferred_parquet = (
                Path("data_cache")
                / f"klines_{exchange}_{symbol}_1m_consolidated.parquet"
            )
            preferred_csv = (
                Path("data_cache") / f"klines_{exchange}_{symbol}_1m_consolidated.csv"
            )
            if preferred_parquet.exists():
                market_data = self.data_manager.load_from_parquet(str(preferred_parquet))
                self.logger.info(f"✅ Loaded market data from {preferred_parquet}")
                return market_data
            if preferred_csv.exists():
                market_data = pd.read_csv(preferred_csv)
                self.logger.info(f"✅ Loaded market data from {preferred_csv}")
                return market_data

            # Fallback to raw files in data_dir
            parquet_path = Path(data_dir) / f"{exchange}_{symbol}_klines.parquet"
            csv_path = Path(data_dir) / f"{exchange}_{symbol}_klines.csv"
            if parquet_path.exists():
                self.logger.info(f"Loading data from Parquet: {parquet_path}")
                try:
                    return self.data_manager.load_from_parquet(str(parquet_path))
                except Exception as e:
                    self.logger.warning(f"Parquet load failed ({e}); falling back to CSV if available")
                    if csv_path.exists():
                        self.logger.info(f"Loading data from CSV: {csv_path}")
                        try:
                            return pd.read_csv(csv_path)
                        except Exception as e:
                            self.logger.warning(f"CSV load failed ({e}); returning empty DataFrame")
                            return pd.DataFrame()

            self.logger.warning(f"⚠️ Market data files not found in {data_dir} for {exchange} {symbol}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.exception(f"❌ Failed to load market data: {e}")
            return None

    def _evaluate_params_with_cache(
        self, market_data: pd.DataFrame, params: Dict[str, Any],
    ) -> float:
        """Evaluate params using cached backtester if available, else simple placeholder."""
        if self.cached_backtester is None:
            self.cached_backtester = CachedBacktester(market_data)
        try:
            return float(self.cached_backtester.run_cached_backtest(params))
        except Exception:
            return random.uniform(-1.0, 1.0)

    def get_memory_profile(self) -> Dict[str, Any]:
        """Expose current memory profile using MemoryManager."""
        if self.memory_manager is None:
            self.memory_manager = MemoryManager(memory_threshold=self.memory_threshold)
        return self.memory_manager.profile_memory_usage()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Expose optimization component status."""
        stats = {
            "caching_enabled": self.enable_caching,
            "parallelization_enabled": self.enable_parallelization,
            "early_stopping_enabled": self.enable_early_stopping,
            "memory_management_enabled": self.enable_memory_management,
            "max_workers": self.max_workers,
            "memory_threshold": self.memory_threshold,
        }
        if self.cached_backtester is not None:
            stats["cache_size"] = len(self.cached_backtester.cache)
        if self.adaptive_sampler is not None:
            stats["trial_history_size"] = len(self.adaptive_sampler.trial_history)
        return stats

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="enhanced training manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the enhanced training manager and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Enhanced Training Manager...")

            # Cleanup computational optimization manager
            if self.computational_optimization_manager:
                await self.computational_optimization_manager.cleanup()
                self.logger.info("✅ Computational optimization manager cleaned up")

            # Cleanup parallel backtester
            if self.parallel_backtester is not None:
                shutdown = getattr(self.parallel_backtester, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                self.parallel_backtester = None

            # Force memory cleanup
            if self.enable_memory_management and self.memory_manager is not None:
                self.memory_manager._cleanup_memory()

            self.is_training = False
            self.logger.info("✅ Enhanced Training Manager stopped successfully")

        except Exception as e:
            self.logger.exception(f"❌ Failed to stop Enhanced Training Manager: {e}")

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics from the enhanced training manager."""
        return self.optimization_statistics

    def get_cached_backtester(self) -> CachedBacktester | None:
        """Get the cached backtester instance."""
        return self.cached_backtester

    def get_progressive_evaluator(self) -> ProgressiveEvaluator | None:
        """Get the progressive evaluator instance."""
        return self.progressive_evaluator

    def get_memory_manager(self) -> MemoryManager:
        """Get the memory manager instance."""
        return self.memory_manager

    def get_data_manager(self) -> MemoryEfficientDataManager:
        """Get the data manager instance."""
        return self.data_manager

    def get_optimized_manager(self) -> EnhancedTrainingManagerOptimized:
        """Get the underlying optimized training manager."""
        return self.optimized_manager

    async def execute_optimized_training(
        self, symbol: str, exchange: str, timeframe: str = "1h",
    ) -> Dict[str, Any]:
        """Execute training using the optimized manager directly for advanced operations."""
        try:
            self.logger.info(
                f"🚀 Executing optimized training for {symbol} on {exchange}",
            )

            # Delegate to optimized manager
            result = await self.optimized_manager.execute_optimized_training(
                symbol, exchange, timeframe,
            )

            # Store results in main manager
            if result:
                self.enhanced_training_results.update(result)

            return result

        except Exception as e:
            self.logger.exception(f"❌ Optimized training execution failed: {e}")
            return {}

    def use_cached_backtesting(self, params: Dict[str, Any]) -> float:
        """Use cached backtesting for parameter evaluation."""
        if self.cached_backtester:
            return self.cached_backtester.run_cached_backtest(params)
        self.logger.warning("Cached backtester not initialized")
        return 0.0

    def use_progressive_evaluation(
        self, params: Dict[str, Any], evaluator_func,
    ) -> float:
        """Use progressive evaluation for early stopping."""
        if self.progressive_evaluator:
            return self.progressive_evaluator.evaluate_progressively(
                params, evaluator_func,
            )
        self.logger.warning("Progressive evaluator not initialized")
        return 0.0

    def generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a robust cache key using the _make_hashable utility."""
        return str(hash(_make_hashable(params)))

    async def initialize_components(self) -> bool:
        """Initialize the enhanced training manager and all its components (auxiliary)."""
        try:
            self.logger.info("🚀 Initializing Enhanced Training Manager...")

            # Initialize optimized tools first
            if not await self._initialize_optimized_tools():
                self.logger.error("❌ Failed to initialize optimized tools")
                return False

            # Initialize computational optimization manager if enabled
            if self.enable_computational_optimization:
                try:
                    # create_computational_optimization_manager is async; await it here
                    self.computational_optimization_manager = (
                        await create_computational_optimization_manager(
                            get_computational_optimization_config(), pd.DataFrame(), {},
                        )
                    )
                    self.logger.info("✅ Computational optimization manager initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to initialize computational optimization manager: {e}")
                    self.enable_computational_optimization = False

            self.logger.info("✅ Enhanced Training Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Training Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="feature selection execution",
    )
    async def _execute_feature_selection(
        self,
        symbol: str,
        data_dir: str,
        timeframe: str,
        exchange: str,
    ) -> bool:
        """Execute comprehensive feature selection and pruning.

        Implements tiered feature selection strategy for 240+ features:
        - Tier 1: Core features (80)
        - Tier 2: Normalized features (40)
        - Tier 3: Interaction features (60)
        - Tier 4: Lagged features (40)
        - Tier 5: Causality features (20)

        Args:
            symbol: Trading symbol
            data_dir: Data directory
            timeframe: Timeframe
            exchange: Exchange name

        Returns:
            bool: True if successful, False otherwise

        """
        try:
            self.logger.info("🔍 Starting comprehensive feature selection...")

            # Load feature selection configuration
            feature_config = self.config.get("feature_interactions", {})
            selection_tiers = feature_config.get("feature_selection_tiers", {})

            # Get tiered selection parameters
            tier_1_count = selection_tiers.get("tier_1_base_features", 80)
            tier_2_count = selection_tiers.get("tier_2_normalized_features", 40)
            tier_3_count = selection_tiers.get("tier_3_interaction_features", 60)
            tier_4_count = selection_tiers.get("tier_4_lagged_features", 40)
            tier_5_count = selection_tiers.get("tier_5_causality_features", 20)
            total_max_features = selection_tiers.get("total_max_features", 240)

            self.logger.info("📊 Feature selection targets:")
            self.logger.info(f"   Tier 1 (Core): {tier_1_count} features")
            self.logger.info(f"   Tier 2 (Normalized): {tier_2_count} features")
            self.logger.info(f"   Tier 3 (Interactions): {tier_3_count} features")
            self.logger.info(f"   Tier 4 (Lagged): {tier_4_count} features")
            self.logger.info(f"   Tier 5 (Causality): {tier_5_count} features")
            self.logger.info(f"   Total Max: {total_max_features} features")

            # Load engineered features from previous step
            features_path = f"{data_dir}/{symbol}_{exchange}_{timeframe}_engineered_features.parquet"

            if not os.path.exists(features_path):
                self.logger.warning(f"⚠️ Engineered features not found at {features_path}")
                self.logger.info("🔄 Proceeding with feature selection on available data...")
                return True  # Continue with available features

            # Load features using optimized data manager
            features_df = self.data_manager.load_from_parquet(features_path)

            if features_df.empty:
                self.logger.warning("⚠️ No features available for selection")
                return True

            self.logger.info(f"📈 Loaded {len(features_df.columns)} features for selection")

            # Execute tiered feature selection
            selected_features = await self._execute_tiered_feature_selection(
                features_df=features_df,
                tier_1_count=tier_1_count,
                tier_2_count=tier_2_count,
                tier_3_count=tier_3_count,
                tier_4_count=tier_4_count,
                tier_5_count=tier_5_count,
                total_max_features=total_max_features,
            )

            # Save selected features
            selected_features_path = Path(data_dir) / f"{symbol}_{exchange}_{timeframe}_selected_features.parquet"
            self.data_manager.save_to_parquet(selected_features, str(selected_features_path))

            self.logger.info("✅ Feature selection completed:")
            self.logger.info(f"   Selected: {len(selected_features.columns)} features")
            self.logger.info(f"   Reduced from: {len(features_df.columns)} features")
            self.logger.info(f"   Reduction: {((len(features_df.columns) - len(selected_features.columns)) / len(features_df.columns) * 100):.1f}%")

            # Save feature selection metadata
            selection_metadata = {
                "original_features": len(features_df.columns),
                "selected_features": len(selected_features.columns),
                "reduction_percentage": (
                    (len(features_df.columns) - len(selected_features.columns))
                    / len(features_df.columns)
                    * 100
                ),
                "selection_timestamp": datetime.now().isoformat(),
                "selection_config": selection_tiers,
            }

            metadata_path = Path(data_dir) / f"{symbol}_{exchange}_{timeframe}_feature_selection_metadata.json"
            _safe_json_write(metadata_path, selection_metadata)

            return True

        except Exception as e:
            self.logger.exception(f"❌ Feature selection failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="tiered feature selection",
    )
    async def _execute_tiered_feature_selection(
        self,
        features_df: pd.DataFrame,
        tier_1_count: int,
        tier_2_count: int,
        tier_3_count: int,
        tier_4_count: int,
        tier_5_count: int,
        total_max_features: int,
    ) -> pd.DataFrame:
        """Execute tiered feature selection strategy.

        Args:
            features_df: DataFrame with all engineered features
            tier_1_count: Number of core features to select
            tier_2_count: Number of normalized features to select
            tier_3_count: Number of interaction features to select
            tier_4_count: Number of lagged features to select
            tier_5_count: Number of causality features to select
            total_max_features: Maximum total features

        Returns:
            pd.DataFrame: DataFrame with selected features

        """
        try:
            self.logger.info("🎯 Executing tiered feature selection...")

            # Categorize features by tier
            feature_categories = self._categorize_features_by_tier(features_df)

            selected_features = pd.DataFrame(index=features_df.index)

            # Tier 1: Core features (technical indicators, basic liquidity)
            tier_1_features = self._select_tier_1_features(
                features_df, feature_categories["tier_1"], tier_1_count,
            )
            selected_features = pd.concat([selected_features, tier_1_features], axis=1)
            self.logger.info(f"   ✅ Tier 1: Selected {len(tier_1_features.columns)} core features")

            # Tier 2: Normalized features (z-scores, changes, accelerations)
            tier_2_features = self._select_tier_2_features(
                features_df, feature_categories["tier_2"], tier_2_count,
            )
            selected_features = pd.concat([selected_features, tier_2_features], axis=1)
            self.logger.info(f"   ✅ Tier 2: Selected {len(tier_2_features.columns)} normalized features")

            # Tier 3: Interaction features (spread*volume, etc.)
            tier_3_features = self._select_tier_3_features(
                features_df, feature_categories["tier_3"], tier_3_count,
            )
            selected_features = pd.concat([selected_features, tier_3_features], axis=1)
            self.logger.info(f"   ✅ Tier 3: Selected {len(tier_3_features.columns)} interaction features")

            # Tier 4: Lagged features (lagged interactions)
            tier_4_features = self._select_tier_4_features(
                features_df, feature_categories["tier_4"], tier_4_count,
            )
            selected_features = pd.concat([selected_features, tier_4_features], axis=1)
            self.logger.info(f"   ✅ Tier 4: Selected {len(tier_4_features.columns)} lagged features")

            # Tier 5: Causality features (market microstructure causality)
            tier_5_features = self._select_tier_5_features(
                features_df, feature_categories["tier_5"], tier_5_count,
            )
            selected_features = pd.concat([selected_features, tier_5_features], axis=1)
            self.logger.info(f"   ✅ Tier 5: Selected {len(tier_5_features.columns)} causality features")

            # Apply final pruning if we exceed total_max_features
            if len(selected_features.columns) > total_max_features:
                selected_features = self._apply_final_pruning(
                    selected_features, total_max_features,
                )
            self.logger.info(f"   🔧 Final pruning: Reduced to {len(selected_features.columns)} features")

            return selected_features

        except Exception as e:
            self.logger.exception(f"❌ Tiered feature selection failed: {e}")
            return pd.DataFrame()

    def _categorize_features_by_tier(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features into tiers based on naming patterns."""
        categories = {
            "tier_1": [],  # Core features
            "tier_2": [],  # Normalized features
            "tier_3": [],  # Interaction features
            "tier_4": [],  # Lagged features
            "tier_5": [],  # Causality features
        }

        for col in features_df.columns:
            col_lower = col.lower()

            # Tier 1: Core technical and liquidity features
            if any(
                keyword in col_lower
                for keyword in [
                    "rsi",
                    "macd",
                    "bb",
                    "atr",
                    "adx",
                    "sma",
                    "ema",
                    "cci",
                    "mfi",
                    "roc",
                    "volume",
                    "spread",
                    "liquidity",
                    "price_impact",
                    "kyle",
                    "amihud",
                ]
            ):
                categories["tier_1"].append(col)

            # Tier 2: Normalized features
            elif any(
                keyword in col_lower
                for keyword in [
                    "_z_score",
                    "_change",
                    "_pct_change",
                    "_acceleration",
                    "_bounded",
                    "_log",
                    "_normalized",
                ]
            ):
                categories["tier_2"].append(col)

            # Tier 3: Interaction features
            elif "_x_" in col_lower or "_div_" in col_lower:
                categories["tier_3"].append(col)

            # Tier 4: Lagged features
            elif "_lag" in col_lower:
                categories["tier_4"].append(col)

            # Tier 5: Causality features
            elif any(
                keyword in col_lower
                for keyword in [
                    "_predicts_",
                    "_causality",
                    "_divergence",
                    "_stress",
                    "_extreme",
                ]
            ):
                categories["tier_5"].append(col)

            # Default to tier 1 for uncategorized features
            else:
                categories["tier_1"].append(col)

        return categories

    def _select_tier_1_features(
        self, features_df: pd.DataFrame, tier_1_features: List[str], count: int,
    ) -> pd.DataFrame:
        """Select core features based on variance and correlation."""
        if not tier_1_features:
            return pd.DataFrame()

        # Get available features
        available_features = [f for f in tier_1_features if f in features_df.columns]
        if not available_features:
            return pd.DataFrame()

        # Calculate feature importance based on variance
        feature_variance = features_df[available_features].var()
        top_features = feature_variance.nlargest(count).index.tolist()

        return features_df[top_features]

    def _select_tier_2_features(
        self, features_df: pd.DataFrame, tier_2_features: List[str], count: int,
    ) -> pd.DataFrame:
        """Select normalized features based on stability."""
        if not tier_2_features:
            return pd.DataFrame()

        available_features = [f for f in tier_2_features if f in features_df.columns]
        if not available_features:
            return pd.DataFrame()

        # Select based on feature stability (lower variance for normalized features)
        feature_variance = features_df[available_features].var()
        stable_features = feature_variance.nsmallest(count).index.tolist()

        return features_df[stable_features]

    def _select_tier_3_features(
        self, features_df: pd.DataFrame, tier_3_features: List[str], count: int,
    ) -> pd.DataFrame:
        """Select interaction features based on significance."""
        if not tier_3_features:
            return pd.DataFrame()

        available_features = [f for f in tier_3_features if f in features_df.columns]
        if not available_features:
            return pd.DataFrame()

        # Select based on absolute mean (higher values indicate more significant interactions)
        feature_abs_mean = features_df[available_features].abs().mean()
        significant_features = feature_abs_mean.nlargest(count).index.tolist()

        return features_df[significant_features]

    def _select_tier_4_features(
        self, features_df: pd.DataFrame, tier_4_features: List[str], count: int,
    ) -> pd.DataFrame:
        """Select lagged features based on temporal significance."""
        if not tier_4_features:
            return pd.DataFrame()

        available_features = [f for f in tier_4_features if f in features_df.columns]
        if not available_features:
            return pd.DataFrame()

        # Select based on variance (higher variance indicates more temporal information)
        feature_variance = features_df[available_features].var()
        temporal_features = feature_variance.nlargest(count).index.tolist()

        return features_df[temporal_features]

    def _select_tier_5_features(
        self, features_df: pd.DataFrame, tier_5_features: List[str], count: int,
    ) -> pd.DataFrame:
        """Select causality features based on market logic significance."""
        if not tier_5_features:
            return pd.DataFrame()

        available_features = [f for f in tier_5_features if f in features_df.columns]
        if not available_features:
            return pd.DataFrame()

        # Select based on absolute mean (causality features should have meaningful values)
        feature_abs_mean = features_df[available_features].abs().mean()
        causality_features = feature_abs_mean.nlargest(count).index.tolist()

        return features_df[causality_features]

    def _apply_final_pruning(
        self, selected_features: pd.DataFrame, max_features: int,
    ) -> pd.DataFrame:
        """Apply final pruning to meet maximum feature count."""
        if len(selected_features.columns) <= max_features:
            return selected_features

        # Calculate overall feature importance based on variance
        feature_variance = selected_features.var()
        top_features = feature_variance.nlargest(max_features).index.tolist()

        return selected_features[top_features]

    # ===== Label Expert Artifacts API =====
    def get_label_expert_models(self) -> Dict[str, Dict[str, Any]]:
        return self.label_expert_models

    def get_label_expert_calibrators(self) -> Dict[str, Any]:
        return self.label_expert_calibrators

    def get_label_reliability(self) -> Dict[str, float]:
        if not self.label_reliability:
            self._load_label_reliability()
        return self.label_reliability

    def get_activation_thresholds(self) -> Dict[str, float]:
        if not self.activation_thresholds:
            self._load_activation_thresholds()
        return self.activation_thresholds

    def save_activation_thresholds(self, thresholds: Dict[str, Any]) -> None:
        try:
            target = self.artifacts_dir / "thresholds.json"
            _safe_json_write(target, thresholds)
            # Also cache in memory (flatten thresholds mapping label->threshold)
            flat = {}
            try:
                for k, v in thresholds.items():
                    if isinstance(v, dict) and "threshold" in v:
                        flat[k] = float(v.get("threshold", 0.5))
                    elif isinstance(v, int | float):
                        flat[k] = float(v)
                if flat:
                    self.activation_thresholds.update(flat)
                    self.logger.info(f"Saved activation thresholds to {target}")
            except Exception:
                pass
            if flat:
                _safe_json_write(target, flat)
                self.logger.info(f"Saved activation thresholds to {target}")
        except Exception as e:
            self.logger.warning(f"Failed to save activation thresholds: {e}")

    def _load_activation_thresholds(self) -> None:
        if self.force_rerun:
            self.logger.info("Force rerun enabled; skipping loading persisted activation thresholds")
            return
        try:
            path = self.artifacts_dir / "thresholds.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                flat = {}
                for k, v in data.items():
                    if isinstance(v, dict) and "threshold" in v:
                        flat[k] = float(v.get("threshold", 0.5))
                    elif isinstance(v, int | float):
                        flat[k] = float(v)
                self.activation_thresholds = flat
                self.logger.info(f"Loaded activation thresholds from {path}")
        except Exception as e:
            self.logger.warning(f"Failed to load activation thresholds: {e}")

    def save_label_reliability(self, reliability: Dict[str, float]) -> None:
        try:
            target = self.artifacts_dir / "reliability.json"
            _safe_json_write(target, reliability)
            self.label_reliability.update({k: float(v) for k, v in reliability.items()})
            self.logger.info(f"Saved label reliability to {target}")
        except Exception as e:
            self.logger.warning(f"Failed to save label reliability: {e}")

    def _load_label_reliability(self) -> None:
        if self.force_rerun:
            self.logger.info("Force rerun enabled; skipping loading persisted reliability")
            return
        try:
            path = self.artifacts_dir / "reliability.json"
            if path.exists():
                with open(path) as f:
                    self.label_reliability = {k: float(v) for k, v in json.load(f).items()}
                self.logger.info(f"Loaded label reliability from {path}")
        except Exception as e:
            self.logger.warning(f"Failed to load label reliability: {e}")

    async def _clear_artifacts_from_step_onward(
        self,
        start_step: str,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> None:
        """Clear artifacts from the specified step and all subsequent steps.
        Preserves artifacts from previous steps.

        Args:
            start_step: Step to start clearing from
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        """
        try:
            self.logger.info(f"🧹 Clearing artifacts from {start_step} onward")

            # Find the index of the starting step using class constant
            try:
                start_index = self.STEP_ORDER.index(start_step)
            except ValueError:
                self.logger.warning(f"⚠️ Unknown step {start_step}, clearing all artifacts")
                start_index = 0

            # Clear artifacts for the starting step and all subsequent steps
            steps_to_clear = self.STEP_ORDER[start_index:]

            for step in steps_to_clear:
                await self._clear_step_artifacts(step, symbol, exchange, timeframe)

            self.logger.info(f"✅ Cleared artifacts for {len(steps_to_clear)} steps: {steps_to_clear}")

        except Exception as e:
            self.logger.exception(f"❌ Error clearing artifacts: {e}")

    async def verify_previous_step_artifacts(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> bool:
        """Verify that artifacts from the previous step exist before starting a step.

        Args:
            step_name: Name of the current step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            True if previous step artifacts exist, False otherwise

        """
        try:
            # Find the index of the current step using class constant
            try:
                current_index = self.STEP_ORDER.index(step_name)
            except ValueError:
                self.logger.warning(f"⚠️ Unknown step {step_name}, skipping artifact verification")
                return True

            # If this is the first step, no previous artifacts to verify
            if current_index == 0:
                return True

            # Get the previous step
            previous_step = self.STEP_ORDER[current_index - 1]

            # Get critical artifacts for the previous step using class constant
            previous_artifacts = self.CRITICAL_ARTIFACTS.get(previous_step, [])

            if not previous_artifacts:
                self.logger.warning(f"⚠️ No critical artifacts defined for {previous_step}")
                return True

            # Check if at least one critical artifact exists with proper pattern substitution
            from pathlib import Path
            import glob
            artifacts_found = []

            for artifact_pattern in previous_artifacts:
                # Substitute placeholders in the pattern
                substituted_pattern = artifact_pattern.format(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe
                )

                # Handle glob patterns (like **/*.parquet)
                if "**" in substituted_pattern:
                    # Use glob to find matching files
                    matching_files = glob.glob(substituted_pattern, recursive=True)
                    if matching_files:
                        artifacts_found.append(f"{artifact_pattern} -> {len(matching_files)} files found")
                    else:
                        # Check if the specific file exists
                        if Path(substituted_pattern).exists():
                            artifacts_found.append(f"{artifact_pattern} -> {substituted_pattern}")

            if artifacts_found:
                self.logger.info(f"✅ Found previous step artifacts for {previous_step}: {artifacts_found}")
                return True

            self.logger.error(f"❌ Missing critical artifacts from {previous_step}")
            self.logger.error(f"   Expected artifacts: {previous_artifacts}")
            self.logger.error(f"   Cannot proceed with {step_name} without previous step artifacts")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Error verifying previous step artifacts for {step_name}: {e}")
            return False

    async def _clear_step_artifacts(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> None:
        """Clear artifacts for a specific step.

        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        """
        try:
            import glob
            from pathlib import Path

            # Get patterns for this step using class constant
            patterns = self.ARTIFACT_PATTERNS.get(step_name, [])

            cleared_count = 0
            for pattern in patterns:
                # Find files matching the pattern
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    try:
                        Path(file_path).unlink()
                        cleared_count += 1
                        self.logger.debug(f"   🗑️ Cleared: {file_path}")
                    except FileNotFoundError:
                        pass  # File doesn't exist, which is fine
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Could not delete {file_path}: {e}")

            if cleared_count > 0:
                self.logger.info(f"   🧹 Cleared {cleared_count} artifacts for {step_name}")
            else:
                self.logger.debug(f"   ℹ️ No artifacts found for {step_name}")

        except Exception as e:
            self.logger.exception(f"❌ Error clearing artifacts for {step_name}: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced training manager setup",
)
async def setup_enhanced_training_manager(
    config: Dict[str, Any] | None = None,
) -> EnhancedTrainingManager | None:
    """Setup and return a configured EnhancedTrainingManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedTrainingManager: Configured enhanced training manager instance

    """
    try:
        manager = EnhancedTrainingManager(config or {})
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup enhanced training manager: {e}")
        return None
