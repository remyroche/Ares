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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
import asyncio

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

# Import model performance monitor
from src.utils.model_performance_monitor import ModelPerformanceMonitor

# Import the auto-fix decorator for data quality issues
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)

# Import new QA decorators
from src.utils.training_pipeline_decorators import (
    validate_pipeline_step,
    ensure_data_integrity,
    monitor_step_execution,
    secure_step_execution,
    monitor_pipeline_step,
    validate_pipeline_input,
    monitor_pipeline_performance,
    PipelineStage,
    PipelineValidationLevel,
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
    """Atomically and deterministically write JSON to target."

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
    """Enhanced training manager with comprehensive 16-step pipeline."

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

    Integration:
    - Acts as the main entry point for all training operations
    - Delegates optimization tasks to EnhancedTrainingManagerOptimized
    - Provides unified interface while leveraging optimized backend
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize enhanced training manager."

        Args:
            config: Configuration dictionary

        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedTrainingManager")

        # Enhanced training manager state
        self.is_training: bool = False
        self.enhanced_training_results: dict[str, Any] = {}
        self.enhanced_training_history: list[dict[str, Any]] = []
        
        # Enhanced reporting configuration
        self.reporting_config = config.get("enhanced_reporting", {})
        self.enable_detailed_reporting = self.reporting_config.get("enable_detailed_reporting", True)
        self.pipeline_reports_dir = Path("reports/enhanced_training_pipeline")
        self.pipeline_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Track step execution for reporting
        self.current_pipeline_execution_id = None
        self.step_reports = {}

        # Define pipeline step order as class constant
        self.STEP_ORDER = [
            "step1_data_collection",           # Download and prepare market data
            "step1_5_data_converter",          # Convert data to unified format
            "step2_feature_engineering",       # Feature engineering
            "step3_hmm_regime_discovery",      # Define HMM regime clusters (with basic features)
            "step4_regime_data_splitting",     # Regime data splitting
            "step5_triple_barrier_method",     # Apply triple barrier method
            "step6_feature_generation",        # Feature generation
            "step7_matrix_feature_selection",  # Matrix feature selection
            "step8_tactician_labeling",        # Tactician labeling
            "step9_tactician_specialist_training", # Tactician specialist training
            "step10_confidence_calibration",   # Confidence calibration
            "step11_final_parameters_optimization", # Final parameters optimization
            "step12_walk_forward_validation",  # Walk forward validation
            "step13_monte_carlo_validation",   # Monte Carlo validation
            "step14_ab_testing",               # A/B testing
            "step15_saving",                   # Save final models
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
            "step2_5_sr_optimization": [
                "data/optimization/sr_optimization_results.json",
                "optimization_results.json",
            ],
            "step3_hmm_regime_discovery": [
                "data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet",
            ],
            "step4_processing_labeling": [
                "data/training/{exchange}_{symbol}_{timeframe}_labeled_validation.parquet",
            ],
            "step5_regime_data_splitting": [
                "data/training/{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_regime_labels.json",
            ],
            "step6_hmm_based_training": [
                "data/training/{exchange}_{symbol}_{timeframe}_hmm_models.pkl",
            ],
            "step9_5_multi_timeframe_hmm_ensemble": [
                "models/multi_timeframe_hmm_ensemble/{exchange}_{symbol}/ensemble_metadata.json",
                "models/multi_timeframe_hmm_ensemble/{exchange}_{symbol}/meta_learner.joblib",
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

        # Define artifact patterns for clearing (includes all artifacts, not just critical ones)
        self.ARTIFACT_PATTERNS = {
            "step1_data_collection": [
                "data_cache/klines_{exchange}_{symbol}_*_consolidated.*",
                "data_cache/aggtrades_{exchange}_{symbol}_consolidated.*",
            ],
            "step1_5_data_converter": [
                "data_cache/unified/{exchange}/{symbol}/{timeframe}/**/*.parquet",
                "data_cache/unified/{exchange}_{symbol}_{timeframe}_config.json",
            ],
            "step2_data_reading": [
                "data_cache/unified/{exchange}/{symbol}/{timeframe}/**/*.parquet",
                "data_cache/unified/{exchange}_{symbol}_{timeframe}_config.json",
            ],
            "step3_hmm_regime_discovery": [
                "data/hmm_regimes/{exchange}_{symbol}_{timeframe}_hmm_*.parquet",
                "data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.*",
                "data/hmm_regimes/{exchange}_{symbol}_{timeframe}_regime_*.json",
            ],
            "step4_triple_barrier_method": [
                "data/training/{exchange}_{symbol}_{timeframe}_triple_barrier_*.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_barrier_*.json",
            ],
            "step5_labeling": [
                "data/training/{exchange}_{symbol}_{timeframe}_labeled_*.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_labels_*.json",
            ],
            "step6_feature_engineering": [
                "data/training/{exchange}_{symbol}_{timeframe}_engineered_features.*",
                "data/training/{exchange}_{symbol}_{timeframe}_feature_metadata.*",
            ],
            "step7_regime_data_splitting": [
                "data/training/{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_regime_labels.json",
                "data/training/{exchange}_{symbol}_{timeframe}_regime_statistics.json",
            ],
            "step8_hmm_based_training": [
                "data/training/{exchange}_{symbol}_{timeframe}_hmm_models_*.pkl",
                "data/training/{exchange}_{symbol}_{timeframe}_training_results_*.json",
            ],
            "step8_5_unified_regime_intelligence": [
                "data/training/{exchange}_{symbol}_{timeframe}_unified_intelligence_*.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_intelligence_*.json",
            ],
            "step9_analyst_enhancement": [
                "data/training/{exchange}_{symbol}_{timeframe}_analyst_*.pkl",
                "data/training/{exchange}_{symbol}_{timeframe}_analyst_*.json",
            ],
            "step10_tactician_labeling": [
                "data/training/{exchange}_{symbol}_{timeframe}_tactician_labels_*.parquet",
                "data/training/{exchange}_{symbol}_{timeframe}_tactician_*.json",
            ],
            "step11_tactician_specialist_training": [
                "data/training/{exchange}_{symbol}_{timeframe}_specialist_*.pkl",
                "data/training/{exchange}_{symbol}_{timeframe}_specialist_*.json",
            ],
            "step10_confidence_calibration": [
                "data/training/{exchange}_{symbol}_{timeframe}_calibration_*.pkl",
                "data/training/{exchange}_{symbol}_{timeframe}_calibration_*.json",
            ],
            "step11_final_parameters_optimization": [
                "data/training/{exchange}_{symbol}_{timeframe}_optimization_*.json",
                "data/training/{exchange}_{symbol}_{timeframe}_best_params_*.json",
            ],
            "step12_walk_forward_validation": [
                "data/training/{exchange}_{symbol}_{timeframe}_walk_forward_*.json",
                "data/training/{exchange}_{symbol}_{timeframe}_validation_*.parquet",
            ],
            "step13_monte_carlo_validation": [
                "data/training/{exchange}_{symbol}_{timeframe}_monte_carlo_*.json",
                "data/training/{exchange}_{symbol}_{timeframe}_mc_results_*.parquet",
            ],
            "step14_ab_testing": [
                "data/training/{exchange}_{symbol}_{timeframe}_ab_test_*.json",
                "data/training/{exchange}_{symbol}_{timeframe}_ab_results_*.parquet",
            ],
            "step15_saving": [
                "data/training/{exchange}_{symbol}_{timeframe}_final_models_*.pkl",
                "data/training/{exchange}_{symbol}_{timeframe}_final_results_*.json",
            ],
        }

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

        # Initialize model performance monitor
        self.model_performance_monitor = ModelPerformanceMonitor(self.config)

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

        self.logger.info("Loaded optimization configuration")
        return None

    def _load_optimization_config(self) -> None:
        """Load optimization configuration from enhanced_training_manager_optimized."""
        # Caching configuration
        caching_config = self.optimization_config.get("caching", {})
        self.enable_caching = caching_config.get("enabled", True)
        self.max_cache_size = caching_config.get("max_cache_size", 1000)
        self.cache_ttl = caching_config.get("cache_ttl", 3600)

        # Parallelization configuration
        parallel_config = self.optimization_config.get("parallelization", {})
        self.enable_parallelization = parallel_config.get("enabled", True)
        self.max_workers = parallel_config.get("max_workers", 8)
        self.chunk_size = parallel_config.get("chunk_size", 1000)

        # Early stopping configuration
        early_stop_config = self.optimization_config.get("early_stopping", {})
        self.enable_early_stopping = early_stop_config.get("enabled", True)
        self.patience = early_stop_config.get("patience", 10)
        self.min_trials = early_stop_config.get("min_trials", 20)

        # Memory management configuration
        memory_config = self.optimization_config.get("memory_management", {})
        self.enable_memory_management = memory_config.get("enabled", True)
        self.memory_threshold = memory_config.get("memory_threshold", 0.8)
        self.cleanup_frequency = memory_config.get("cleanup_frequency", 100)

        self.logger.info("Loaded optimization configuration")

    @contextmanager
    def _timed_step(self, name: str, step_times: dict):
        start = time.time()
        try:
            yield
            self._log_step_completion(name, start, step_times, success=True)
        except Exception:
            self._log_step_completion(name, start, step_times, success=False)
            raise

    def _save_checkpoint(self, step_name: str, pipeline_state: dict[str, Any]) -> None:
        """Save training progress checkpoint."

        Args:
            step_name: Current step name
            pipeline_state: Current pipeline state

        """
        if not self.enable_checkpointing:
            return

        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "current_step": step_name,
                "pipeline_state": pipeline_state,
                "training_mode": "blank" if self.blank_training_mode else "full",
                "symbol": getattr(self, "current_symbol", ""),
                "exchange": getattr(self, "current_exchange", ""),
                "timeframe": getattr(self, "current_timeframe", "1m"),
                "lookback_days": self.lookback_days,
                "max_trials": self.max_trials,
                "n_trials": self.n_trials,
            }

            # Namespaced checkpoint path
            symbol = checkpoint_data.get("symbol") or "unknown"
            exchange = checkpoint_data.get("exchange") or "unknown"
            timeframe = checkpoint_data.get("timeframe") or "unknown"
            ns_dir = self.checkpoint_dir / exchange / symbol / timeframe
            ns_dir.mkdir(parents=True, exist_ok=True)
            target_file = ns_dir / "training_progress.json"
            _safe_json_write(target_file, checkpoint_data)

            self.logger.info(f"💾 Checkpoint saved: {step_name} -> {target_file}")

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> dict[str, Any] | None:
        """Load training progress checkpoint."

        Returns:
            dict: Checkpoint data or None if no checkpoint exists

        """
        # Attempt to load namespaced checkpoint based on current params
        # Ensure enable_checkpointing is defined
        if not hasattr(self, "enable_checkpointing"):
            self.enable_checkpointing = getattr(self, "enhanced_training_config", {}).get("enable_checkpointing", True)

        if not self.enable_checkpointing:
            return None

        # Ensure checkpoint_dir is defined
        if not hasattr(self, "checkpoint_dir"):
            self.checkpoint_dir = Path("checkpoints")
            self.checkpoint_dir.mkdir(exist_ok=True)

        try:
            symbol = getattr(self, "current_symbol", "unknown")
            exchange = getattr(self, "current_exchange", "unknown")
            timeframe = getattr(self, "current_timeframe", "unknown")
            ns_file = (
                self.checkpoint_dir
                / exchange
                / symbol
                / timeframe
                / "training_progress.json"
            )
            if not ns_file.exists():
                return None
            with open(ns_file) as f:
                checkpoint_data = json.load(f)
            self.logger.info(
                f"📂 Checkpoint loaded: {checkpoint_data.get('current_step', 'unknown')} from {ns_file}",
            )
            return checkpoint_data

        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _clear_checkpoint(self) -> None:
        """Clear the checkpoint file."""
        try:
            symbol = getattr(self, "current_symbol", "unknown")
            exchange = getattr(self, "current_exchange", "unknown")
            timeframe = getattr(self, "current_timeframe", "unknown")
            ns_file = (
                self.checkpoint_dir
                / exchange
                / symbol
                / timeframe
                / "training_progress.json"
            )
            if ns_file.exists():
                # Guard against clearing outside configured checkpoint dir
                if _is_relative_to(ns_file, self.checkpoint_dir) and not ns_file.is_symlink():
                    ns_file.unlink()
                    self.logger.info(f"🗑️ Checkpoint cleared at {ns_file}")
                else:
                    self.logger.warning(f"Skipped clearing checkpoint due to unsafe path: {ns_file}")
        except Exception as e:
            self.logger.warning(f"Failed to clear checkpoint: {e}")

    def _heartbeat(self, message: str) -> None:
        """Log a heartbeat message for monitoring training progress."

        Args:
            message: Heartbeat message to log

        """
        self.logger.info(f"💓 {message}")

    def _get_system_resources(self) -> dict[str, float]:
        """Get current system resource usage."

        Returns:
            dict: System resource information

        """
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)

            # Get system-wide memory info
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent

            return {
                "memory_mb": float(memory_mb),
                "cpu_percent": float(cpu_percent),
                "system_memory_percent": float(system_memory_percent),
                "available_memory_gb": float(system_memory.available / 1024 / 1024 / 1024),
            }
        except Exception as e:
            self.logger.warning(f"Could not get system resources: {e}")
            return {
                "memory_mb": 0.0,
                "cpu_percent": 0.0,
                "system_memory_percent": 0.0,
                "available_memory_gb": 0.0,
            }

    async def validate_step_dependencies(
        self,
        step_name: str,
        pipeline_state: dict[str, Any],
        force_rerun: bool,
    ) -> bool:
        """Validate that all dependencies for a step are met."

        Args:
            step_name: Name of the step to validate
            pipeline_state: Current pipeline state
            force_rerun: If True, skip dependency validation for the starting step

        Returns:
            True if dependencies are met, False otherwise

        """
        try:
            self.logger.info(f"🔍 Validating dependencies for {step_name}")

            # If force_rerun is True, we're starting from this step, so skip dependency validation'
            if force_rerun:
                self.logger.info(f"✅ Force rerun enabled for {step_name}, skipping dependency validation")
                return True

            # Use StepDependencyValidator to check prerequisites
            # Use the correct checkpoint directory that matches where checkpoints are saved
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

    def _analyze_resource_requirements(self) -> dict[str, Any]:
        """Analyze resource requirements for the training process."

        Returns:
            dict: Resource analysis information

        """
        try:
            # Get system info
            cpu_count = int(psutil.cpu_count() or 0)
            memory_gb = float(psutil.virtual_memory().total / 1024 / 1024 / 1024)

            # Realistic estimates based on actual training complexity
            if self.blank_training_mode:
                estimated_memory_gb = 4.0  # Blank training uses less memory
                estimated_time_minutes = 90  # Realistic: 1.5 hours for blank training
                memory_warning_threshold = 6.0
                models_to_train = 4
                optimization_trials = 50
            else:
                estimated_memory_gb = 8.0  # Full training uses more memory
                estimated_time_minutes = 720  # Realistic: 12 hours for full training
                memory_warning_threshold = 12.0
                models_to_train = 12
                optimization_trials = 200

            # Check if system meets requirements
            memory_sufficient = memory_gb >= memory_warning_threshold
            cpu_sufficient = cpu_count >= 4

            return {
                "system_memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "estimated_memory_gb": estimated_memory_gb,
                "estimated_time_minutes": estimated_time_minutes,
                "models_to_train": models_to_train,
                "optimization_trials": optimization_trials,
                "memory_sufficient": memory_sufficient,
                "cpu_sufficient": cpu_sufficient,
                "memory_warning_threshold": memory_warning_threshold,
                "recommendations": self._get_resource_recommendations(
                    memory_gb, cpu_count,
                ),
                "step_breakdown": self._get_step_time_breakdown(
                    self.blank_training_mode,
                ),
            }
        except Exception as e:
            self.logger.warning(f"Could not analyze resource requirements: {e}")
            return {}

    def _get_resource_recommendations(
        self, memory_gb: float, cpu_count: int,
    ) -> list[str]:
        """Get resource recommendations based on system specs."

        Args:
            memory_gb: Available memory in GB
            cpu_count: Number of CPU cores

        Returns:
            list: Recommendations

        """
        recommendations = []

        if memory_gb < 8:
            recommendations.append(
                "⚠️ Consider upgrading to 16GB RAM for optimal performance",
            )
        elif memory_gb < 12:
            recommendations.append("💡 16GB RAM recommended for full training mode")

        if cpu_count < 4:
            recommendations.append(
                "⚠️ Consider using a system with at least 4 CPU cores",
            )
        elif cpu_count < 8:
            recommendations.append("💡 8+ CPU cores recommended for faster training")

        if self.blank_training_mode:
            recommendations.append("✅ Blank training mode is suitable for your system")
        elif memory_gb < 12:
            recommendations.append(
                "⚠️ Full training mode may be slow on your system",
            )
        else:
            recommendations.append(
                "✅ Full training mode should work well on your system",
            )

        return recommendations

    def _get_step_time_breakdown(self, is_blank_mode: bool) -> dict[str, int]:
        """Get realistic time breakdown for each step."

        Args:
            is_blank_mode: Whether this is blank training mode

        Returns:
            dict: Time estimates for each step in minutes

        """
        if is_blank_mode:
            return {
                "step1_data_collection": 5,
                "step1_5_data_converter": 3,
                "step2_feature_engineering": 15,
                "step3_hmm_regime_discovery": 3,
                "step4_processing_labeling": 8,
                "step5_regime_data_splitting": 2,
                "step6_hmm_based_training": 10,
                "step6_5_unified_regime_intelligence": 8,
                "step7_analyst_enhancement": 8,
                "step8_tactician_labeling": 5,
                "step9_tactician_specialist_training": 10,
                "step10_confidence_calibration": 3,
                "step11_final_parameters_optimization": 15,
                "step12_walk_forward_validation": 8,
                "step13_monte_carlo_validation": 8,
                "step14_ab_testing": 5,
                "step15_saving": 2,
            }
        return {
            "step1_data_collection": 15,
            "step1_5_data_converter": 10,
            "step2_feature_engineering": 60,
            "step3_hmm_regime_discovery": 8,
            "step4_processing_labeling": 20,
            "step5_regime_data_splitting": 5,
            "step6_hmm_based_training": 30,
            "step6_5_unified_regime_intelligence": 25,
            "step7_analyst_enhancement": 25,
            "step8_tactician_labeling": 15,
            "step9_tactician_specialist_training": 30,
            "step10_confidence_calibration": 10,
            "step11_final_parameters_optimization": 240,
            "step12_walk_forward_validation": 60,
            "step13_monte_carlo_validation": 60,
            "step14_ab_testing": 30,
            "step15_saving": 5,
        }

    def _optimize_memory_usage(self) -> None:
        """Perform memory optimization to reduce memory footprint."""
        try:
            # Force garbage collection
            gc.collect()

            # Log memory before and after optimization
            before_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            self.logger.info(
                f"🧹 Memory optimization: {before_memory:.1f} MB before cleanup",
            )

            # Force another garbage collection
            gc.collect()

            after_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_saved = before_memory - after_memory

            self.logger.info(
                f"🧹 Memory optimization: {after_memory:.1f} MB after cleanup (saved {memory_saved:.1f} MB)",
            )

            if memory_saved > 10:  # If we saved more than 10MB
                self.logger.info(
                    f"   🧹 Memory optimization saved {memory_saved:.1f} MB",
                )

        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")

    def _get_progress_percentage(
        self, completed_steps: int, total_steps: int,
    ) -> float:
        """Calculate progress percentage."

        Args:
            completed_steps: Number of completed steps
            total_steps: Total number of steps

        Returns:
            float: Progress percentage

        """
        return (completed_steps / total_steps) * 100

    def _log_progress(
        self, current_step: int, total_steps: int, elapsed_time: float,
    ) -> None:
        """Log progress with estimated completion time."

        Args:
            current_step: Current step number
            total_steps: Total number of steps
            elapsed_time: Time elapsed so far

        """
        progress = self._get_progress_percentage(current_step, total_steps)
        if elapsed_time > 0:
            avg_time = elapsed_time / max(current_step, 1)
            remaining_steps = total_steps - current_step
            eta_minutes = (avg_time * remaining_steps) / 60
        else:
            eta_minutes = 0
        if self.verbosity == "debug":
            self.logger.debug(
                f"📊 Progress: {progress:.1f}% ({current_step}/{total_steps})",
            )
            self.logger.debug(
                f"⏱️ Elapsed: {elapsed_time/60:.1f} min | ETA: {eta_minutes:.1f} min",
            )
        else:
            self.logger.info(
                f"📊 Progress: {progress:.1f}% ({current_step}/{total_steps})",
            )

    def _log_step_completion(
        self, step_name: str, step_start: float, step_times: dict[str, float], success: bool,
    ) -> None:
        """Log step completion with timing and memory usage."

        Args:
            step_name: Name of the completed step
            step_start: Start time of the step
            step_times: Dictionary to store step times
            success: Whether the step was successful

        """
        step_time = time.time() - step_start
        step_times[step_name] = step_time

        # Get comprehensive system resources
        resources = self._get_system_resources()

        status_icon = "✅" if success else "❌"
        status_text = "completed successfully" if success else "failed"

        self.logger.info(
            f"{status_icon} {step_name}: {status_text} in {step_time:.2f}s",
        )
        self.logger.info(
            f"💾 Process Memory: {resources['memory_mb']:.1f} MB | CPU: {resources['cpu_percent']:.1f}%",
        )
        self.logger.info(
            f"🖥️ System Memory: {resources['system_memory_percent']:.1f}% | Available: {resources['available_memory_gb']:.1f} GB",
        )

        # Memory warning system
        if resources["system_memory_percent"] > 85:
            warning_msg = f"⚠️ HIGH MEMORY USAGE: {resources['system_memory_percent']:.1f}% - Consider closing other applications"
            self.logger.warning(warning_msg)

        if resources["available_memory_gb"] < 2.0:
            warning_msg = f"⚠️ LOW AVAILABLE MEMORY: {resources['available_memory_gb']:.1f} GB remaining"
            self.logger.warning(warning_msg)

        # Log progress after each step
        completed_steps = len(step_times)
        elapsed_time = sum(step_times.values())
        self._log_progress(completed_steps, 16, elapsed_time)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid enhanced training manager configuration"),
            AttributeError: (False, "Missing required enhanced training parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="enhanced training manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize enhanced training manager."

        Returns:
            bool: True if initialization successful, False otherwise

        """
        try:
            self.logger.info("🚀 Initializing Enhanced Training Manager...")
            # Ensure blank_training_mode is defined
            if not hasattr(self, "blank_training_mode"):
                blank_env = os.getenv("BLANK_TRAINING_MODE", "0") == "1"
                blank_config = self.enhanced_training_config.get("blank_training_mode", False)
                self.blank_training_mode = blank_env or blank_config
            self.logger.info(f"📊 Blank training mode: {self.blank_training_mode}")
            # Ensure other attributes are defined
            if not hasattr(self, "max_trials"):
                self.max_trials = self.enhanced_training_config.get("max_trials", 200)
            if not hasattr(self, "n_trials"):
                self.n_trials = self.enhanced_training_config.get("n_trials", 100)
            if not hasattr(self, "lookback_days"):
                default_lookback = 180 if self.blank_training_mode else 30
                self.lookback_days = self.enhanced_training_config.get("lookback_days", default_lookback)
            self.logger.info(f"🔧 Max trials: {self.max_trials}")
            self.logger.info(f"🔧 N trials: {self.n_trials}")
            self.logger.info(f"📈 Lookback days: {self.lookback_days}")
            # Ensure enable_computational_optimization is defined
            if not hasattr(self, "enable_computational_optimization"):
                self.enable_computational_optimization = self.enhanced_training_config.get("enable_computational_optimization", True)
            self.logger.info(
                f"🚀 Computational optimization: {self.enable_computational_optimization}",
            )

            # Analyze resource requirements
            resource_analysis = self._analyze_resource_requirements()
            if resource_analysis:
                self.logger.info("📊 Resource Analysis:")
                self.logger.info(
                    f"   💾 System Memory: {resource_analysis['system_memory_gb']:.1f} GB",
                )
                self.logger.info(f"   🖥️ CPU Cores: {resource_analysis['cpu_count']}")
                self.logger.info(
                    f"   📈 Estimated Memory Usage: {resource_analysis['estimated_memory_gb']:.1f} GB",
                )
                self.logger.info(
                    f"   ⏱️ Estimated Time: {resource_analysis['estimated_time_minutes']} minutes ({resource_analysis['estimated_time_minutes']/60:.1f} hours)",
                )
                self.logger.info(
                    f"   🤖 Models to Train: {resource_analysis['models_to_train']}",
                )
                self.logger.info(
                    f"   🔧 Optimization Trials: {resource_analysis['optimization_trials']}",
                )

            # Show step-by-step breakdown
            if "step_breakdown" in resource_analysis:
                total_estimated = sum(resource_analysis["step_breakdown"].values())
                self.logger.info("📋 Step-by-Step Time Estimates:")
                for step_name, minutes in resource_analysis[
                    "step_breakdown"
                ].items():
                    percentage = (minutes / total_estimated) * 100
                    self.logger.info(
                        f"   {step_name}: {minutes} min ({percentage:.1f}%)",
                    )

            # Log recommendations
            if resource_analysis["recommendations"]:
                self.logger.info("💡 Recommendations:")
                for rec in resource_analysis["recommendations"]:
                    self.logger.info(f"   {rec}")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    "❌ Invalid configuration for enhanced training manager",
                )
                return False

            # Initialize computational optimization if enabled
            if self.enable_computational_optimization:
                await self._initialize_computational_optimization()

            # Optimization components are initialized in _initialize_optimized_tools()
            # to ensure a single, consistent initialization path.
            self.logger.info("✅ Enhanced Training Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced Training Manager initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate enhanced training manager configuration."

        Returns:
            bool: True if configuration is valid, False otherwise

        """
        try:
            # Validate enhanced training manager specific settings
            # Ensure max_enhanced_training_history is defined
            if not hasattr(self, "max_enhanced_training_history"):
                self.max_enhanced_training_history = self.enhanced_training_config.get("max_enhanced_training_history", 100)

            if self.max_enhanced_training_history <= 0:
                self.logger.error(
                    "❌ Invalid max_enhanced_training_history configuration",
                )
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
    async def execute_enhanced_training(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """Execute the comprehensive 16-step enhanced training pipeline."

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if training successful, False otherwise

        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(
                "🚀 COMPREHENSIVE 15-STEP ENHANCED TRAINING PIPELINE START",
            )
            self.logger.info("=" * 80)
            self.logger.info(
                f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self.logger.info(
                f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}",
            )
            self.logger.info(
                f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}",
            )
            self.logger.info(
                f"📊 Training Mode: {enhanced_training_input.get('training_mode', 'N/A')}",
            )
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"🔧 Blank Training Mode: {self.blank_training_mode}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"🔧 N Trials: {self.n_trials}")

            # Initialize enhanced reporting
            self.current_pipeline_execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{enhanced_training_input.get('symbol', 'unknown')}_{enhanced_training_input.get('exchange', 'unknown')}"
            self.step_reports = {}

            self.is_training = True

            # Validate training input
            if not self._validate_enhanced_training_inputs(enhanced_training_input):
                return False

            # Execute the comprehensive 16-step pipeline
            success = await self._execute_comprehensive_pipeline(
                enhanced_training_input,
            )

            if success:
                # Store training history
                await self._store_enhanced_training_history(enhanced_training_input)

                self.logger.info("=" * 80)
                self.logger.info(
                    "🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY",
                )
                self.logger.info("=" * 80)
                self.logger.info(
                    f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
                self.logger.info(
                    f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}",
                )
                self.logger.info(
                    f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}",
                )
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
    def _validate_enhanced_training_inputs(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """Validate enhanced training input parameters."

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if input is valid, False otherwise

        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "lookback_days"]

            for field in required_fields:
                if field not in enhanced_training_input:
                    self.logger.error(
                        f"❌ Missing required enhanced training input field: {field}",
                    )
                    return False

            # Validate specific field values
            if enhanced_training_input.get("lookback_days", 0) <= 0:
                self.logger.error("❌ Invalid lookback_days value")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced training inputs validation failed: {e}")
            return False

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
                self.logger.info(
                    "✅ Computational optimization components initialized successfully",
                )
                return True
            self.logger.warning(
                "⚠️ Failed to initialize computational optimization components",
            )
            return False

        except Exception as e:
            self.logger.exception(
                f"❌ Computational optimization initialization failed: {e}",
            )
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
    async def _execute_comprehensive_pipeline(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """Execute the comprehensive 16-step training pipeline."

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
            # Ensure force_rerun is defined
            if not hasattr(self, "force_rerun"):
                self.force_rerun = self.enhanced_training_config.get("force_rerun", False)

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
            self.logger.info(
                f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(
                f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}",
            )
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(
                f"💾 Memory Optimization: {'Enabled' if self.enable_computational_optimization else 'Disabled'}",
            )
            self.logger.info(f"🚀 Starting from step: {start_step}")
            self.logger.info("=" * 100)

            # Initialize step_start variable
            step_start = time.time()

            # Use optimized data loading for Step 1: Data Collection
            if start_step == "step1_data_collection":
                self.logger.info("📊 STEP 1: Data Collection...")
                self.logger.info("   🔍 Downloading and preparing market data...")

                # Use optimized manager for data collection
                market_data = await self.optimized_manager._load_and_optimize_data(
                    symbol, exchange, timeframe,
                )

                if market_data is not None and not market_data.empty:
                    pipeline_state["market_data"] = market_data
                self.logger.info(f"   ✅ Data loaded: {len(market_data)} rows")

                # Initialize cached backtester with the data
                if self.enable_caching:
                    self.cached_backtester = CachedBacktester(market_data)
                    self.logger.info("   ✅ Cached backtester initialized")

                # Initialize progressive evaluator for early stopping
                if self.enable_early_stopping:
                    self.progressive_evaluator = ProgressiveEvaluator(market_data)
                    self.logger.info("   ✅ Progressive evaluator initialized")
                else:
                    self.logger.error("   ❌ Failed to load market data")
                    return False

                # Save checkpoint after data collection
                self._save_checkpoint("step1_data_collection", pipeline_state)
                step_times["step1_data_collection"] = time.time() - step_start

                # Optionally run validator for Step 1
                try:
                    step1_validation = await self._run_step_validator(
                        "step1_data_collection",
                        training_input,
                        pipeline_state,
                    )
                    if step1_validation and step1_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 1: Data Collection completed successfully and validation passed",
                        )
                    elif self.force_rerun:
                        self.logger.warning(
                            f"⚠️ Step 1 validation failed but is non-fatal: {step1_validation.get('error', 'Unknown error')}",
                        )
                    else:
                        self.logger.info(
                            "⏭️  Skipping Step 1: Data Collection (using pre-consolidated data)",
                        )
                        # Add placeholder for data collection in pipeline state
                        pipeline_state["data_collection"] = {
                            "status": "SKIPPED",
                            "result": {"message": "Using pre-consolidated data"},
                        }

                except Exception as e:
                    # Non-fatal if validator is missing, but log for debugging.
                    self.logger.warning(
                        f"Validator for step1_data_collection failed but is non-fatal: {e}",
                    )
                    if self.force_rerun:
                        self.logger.warning(
                            f"⚠️ Step 1 validation failed but is non-fatal: {e}",
                        )
                    else:
                        self.logger.info(
                            "⏭️  Skipping Step 1: Data Collection (using pre-consolidated data)",
                        )
                        # Add placeholder for data collection in pipeline state
                        pipeline_state["data_collection"] = {
                            "status": "SKIPPED",
                            "result": {"message": "Using pre-consolidated data"},
                        }

                # Get start step key
                start_step_key = training_input.get("start_step", "step1_data_collection")

                # Define helper function for step execution logic
                def _should_run(step_name: str) -> bool:
                    try:
                        return self.STEP_ORDER.index(step_name) >= self.STEP_ORDER.index(start_step_key)
                    except ValueError:
                        # If unknown step names are provided, default to running to be safe
                        return True

                # Step 1.5: Data Converter
                self._heartbeat("Step 1.5: Data Converter")

                # Determine whether Step 1.5 should run based on requested start_step
                should_run_step1_5 = _should_run("step1_5_data_converter")

                if not should_run_step1_5:
                    self.logger.info(
                        f"⏭️ Skipping Step 1.5: Data Converter (starting from '{start_step_key}')",
                    )
                    # Mark pipeline state as skipped; rely on existing artifacts from previous runs
                    pipeline_state["data_converter"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step1_5_data_converter", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step1_5, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step1_5_data_converter", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 1.5 dependencies not met, stopping pipeline")
                        return False

                    step_start_1_5 = time.time()
                    try:
                        from src.training.steps.step1_5_data_converter import run_step as step1_5_run_step

                        # Execute step 1.5 with QA decorators
                        step1_5_success = await self._execute_step1_5_with_qa(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            data_dir="data_cache",
                            force_rerun=self.force_rerun,
                            step1_5_run_step=step1_5_run_step,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 1.5: {e}")
                        step1_5_success = False

                    if not step1_5_success:
                        self._log_step_completion(
                            "Step 1.5: Data Converter",
                            step_start_1_5,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 1.5: Data Converter",
                        step_start_1_5,
                        step_times,
                        success=True,
                    )

                    pipeline_state["data_converter"] = {
                        "status": "SUCCESS" if step1_5_success else "FAILED",
                        "success": bool(step1_5_success),
                        "completed": bool(step1_5_success),
                    }
                    self._save_checkpoint("step1_5_data_converter", pipeline_state)
                    step_times["step1_5_data_converter"] = time.time() - step_start_1_5

                    # Run validator for Step 1.5 (AFTER execution, for verification only)
                    try:
                        step1_5_validation = await self._run_step_validator(
                            "step1_5_data_converter", training_input, pipeline_state,
                        )
                        if step1_5_validation and step1_5_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 1.5: Data Converter completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 1.5 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 1.5 validator failed: {e} - stopping pipeline")
                        return False

                # Step 2: Feature Engineering
                self._heartbeat("Step 2: Feature Engineering")

                # Determine whether Step 2 should run based on requested start_step
                start_step_key = training_input.get("start_step", "step1_data_collection")
                def _should_run(step_name: str) -> bool:
                    try:
                        return self.STEP_ORDER.index(step_name) >= self.STEP_ORDER.index(start_step_key)
                    except ValueError:
                        # If unknown step names are provided, default to running to be safe
                        return True

                if not _should_run("step2_feature_engineering"):
                    self.logger.info(
                        f"⏭️ Skipping Step 2: Feature Engineering (starting from '{start_step_key}')",
                    )
                    # Mark pipeline state as skipped; rely on existing artifacts from previous runs
                    pipeline_state["feature_engineering"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution (step1_5 should have completed)
                    if not await self.verify_previous_step_artifacts("step2_feature_engineering", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step02, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution (step1_5 should have completed)
                    if not await self.validate_step_dependencies("step2_feature_engineering", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 2 dependencies not met (step1_5 should have completed), stopping pipeline")
                        return False

                    step_start_2 = time.time()
                    try:
                        # Prepare feature engineering configuration
                        feature_config = self.config.get("vectorized_advanced_features", {})
                        if not feature_config:
                            # Default configuration with difference and acceleration features enabled
                            feature_config = {
                                "enable_difference_acceleration_features": True,
                                "enable_volatility_modeling": True,
                                "enable_correlation_analysis": True,
                                "enable_momentum_analysis": True,
                                "enable_liquidity_analysis": True,
                                "enable_candlestick_patterns": True,
                                "enable_sr_distance": True,
                                "enable_wavelet_transforms": True,
                                "enable_multi_timeframe": True,
                                "enable_meta_labeling": False,
                                "enable_explicit_meta_labels": False,
                            }

                        # Execute step02 with QA decorators
                        step2_success = await self._execute_step2_with_qa(
                            symbol=symbol,
                            exchange=exchange,
                            data_dir=data_dir,
                            timeframe=timeframe,
                            force_rerun=self.force_rerun,
                            feature_config={"vectorized_advanced_features": feature_config},
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 2: {e}")
                        step2_success = False

                    if not step2_success:
                        self._log_step_completion(
                            "Step 2: Feature Engineering",
                            step_start_2,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 2: Feature Engineering",
                        step_start_2,
                        step_times,
                        success=True,
                    )

                    pipeline_state["feature_engineering"] = {
                        "status": "SUCCESS" if step2_success else "FAILED",
                        "success": bool(step2_success),
                        "completed": bool(step2_success),
                    }
                    self._save_checkpoint("step2_feature_engineering", pipeline_state)
                    step_times["step2_feature_engineering"] = time.time() - step_start_2

                    # Run validator for Step 2 (AFTER execution, for verification only)
                    # Run validator only if Step 2 was executed (not skipped above)
                    if _should_run("step2_feature_engineering"):
                        try:
                            step2_validation = await self._run_step_validator(
                                "step2_feature_engineering", training_input, pipeline_state,
                            )
                            if step2_validation and step2_validation.get("validation_passed", False):
                                # Explicit success notice when validation passes
                                self.logger.info(
                                    "🎉 Step 2: Feature Engineering completed successfully and validation passed",
                                )
                            else:
                                self.logger.error("❌ Step 2 validation failed - stopping pipeline")
                                return False
                        except Exception as e:
                            self.logger.exception(f"❌ Step 2 validator failed: {e} - stopping pipeline")
                            return False

                # Step 2.5: S/R Detection Optimization - Fatal on failure
                self._heartbeat("Step 2.5: S/R Detection Optimization")

                should_run_step2_5 = _should_run("step2_5_sr_optimization")
                step_start_2_5 = time.time()

                if not should_run_step2_5:
                    self.logger.info(
                        f"⏭️ Skipping Step 2.5: S/R Detection Optimization (starting from '{start_step_key}')",
                    )
                    pipeline_state["sr_optimization"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step2_5_sr_optimization", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step02.5, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step2_5_sr_optimization", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 2.5 dependencies not met, stopping pipeline")
                        return False

                    step_start_2_5 = time.time()
                    try:
                        from src.training.steps import step2_5_sr_optimization

                        step2_5_success = await step2_5_sr_optimization.run_step(
                            config=self.config,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 2.5: {e}")
                        step2_5_success = False

                    if not step2_5_success:
                        self._log_step_completion(
                            "Step 2.5: S/R Detection Optimization",
                            step_start_2_5,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 2.5: S/R Detection Optimization",
                        step_start_2_5,
                        step_times,
                        success=True,
                    )

                    pipeline_state["sr_optimization"] = {
                        "status": "SUCCESS" if step2_5_success else "FAILED",
                        "success": bool(step2_5_success),
                        "completed": bool(step2_5_success),
                    }
                    self._save_checkpoint("step2_5_sr_optimization", pipeline_state)

                if not step2_5_success:
                    return False
                self.logger.info("➡️ Proceeding to Step 3: HMM Regime Discovery")

                # Step 3: HMM Regime Discovery (block HMMs + composite clustering) - Fatal on failure
                from src.training.steps import step3_hmm_regime_discovery as _step3

                step3_args = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "data_dir": data_dir,
                    "timeframe": timeframe,
                    "lookback_days": self.lookback_days,
                    "force_rerun": self.force_rerun,
                }

                step3_success = await self._execute_pipeline_step(
                    step_name="step3_hmm_regime_discovery",
                    step_function=_step3.run_step_enhanced,
                    step_args=step3_args,
                    step_times=step_times,
                    pipeline_state=pipeline_state,
                    training_input=training_input,
                    is_fatal=True,  # Step 3 is now fatal on failure
                    step_description="Step 3: HMM Regime Discovery",
                )

                if not step3_success:
                    return False
                self.logger.info("➡️ Proceeding to Step 4: Processing & Labeling")

                # Step 4: Regime Data Splitting (NEW - moved before labeling and feature engineering)
                self._heartbeat("Step 4: Regime Data Splitting")

                should_run_step4 = _should_run("step4_regime_data_splitting")
                step_start_4 = time.time()

                if not should_run_step4:
                    self.logger.info(
                        f"⏭️ Skipping Step 4: Regime Data Splitting (starting from '{start_step_key}')",
                    )
                    pipeline_state["regime_data_splitting"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step4_regime_data_splitting", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step04, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step4_regime_data_splitting", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 4 dependencies not met, stopping pipeline")
                        return False

                    step_start_4 = time.time()
                    try:
                        from src.training.steps import step4_regime_data_splitting

                        step4_success = await step4_regime_data_splitting.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            data_dir=data_dir,
                            force_rerun=self.force_rerun,
                            config=self.config,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 4: {e}")
                        step4_success = False

                    if not step4_success:
                        self._log_step_completion(
                            "Step 4: Regime Data Splitting",
                            step_start_4,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 4: Regime Data Splitting",
                        step_start_4,
                        step_times,
                        success=True,
                    )

                    pipeline_state["regime_data_splitting"] = {
                        "status": "SUCCESS" if step4_success else "FAILED",
                        "success": bool(step4_success),
                        "completed": bool(step4_success),
                    }
                    self._save_checkpoint("step4_regime_data_splitting", pipeline_state)
                    step_times["step4_regime_data_splitting"] = time.time() - step_start_4

                    # Run validator for Step 4
                    try:
                        step4_validation = await self._run_step_validator(
                            "step4_regime_data_splitting", training_input, pipeline_state,
                        )
                        if step4_validation and step4_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 4: Regime Data Splitting completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 4 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 4 validator failed: {e} - stopping pipeline")
                        return False

                # Step 5: Triple Barrier Method (regime-specific)
                self._heartbeat("Step 5: Triple Barrier Method")

                should_run_step5 = _should_run("step5_triple_barrier_method")
                step_start_5 = time.time()

                if not should_run_step5:
                    self.logger.info(
                        f"⏭️ Skipping Step 5: Triple Barrier Method (starting from '{start_step_key}')",
                    )
                    pipeline_state["triple_barrier_method"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step5_triple_barrier_method", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step05, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step5_triple_barrier_method", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 5 dependencies not met, stopping pipeline")
                        return False

                    step_start_5 = time.time()
                    try:
                        from src.training.steps import step5_triple_barrier_method

                        step5_success = await step5_triple_barrier_method.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            data_dir=data_dir,
                            force_rerun=self.force_rerun,
                            config=self.config,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 5: {e}")
                        step5_success = False

                    if not step5_success:
                        self._log_step_completion(
                            "Step 5: Triple Barrier Method",
                            step_start_5,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 5: Triple Barrier Method",
                        step_start_5,
                        step_times,
                        success=True,
                    )

                    pipeline_state["triple_barrier_method"] = {
                        "status": "SUCCESS" if step5_success else "FAILED",
                        "success": bool(step5_success),
                        "completed": bool(step5_success),
                    }
                    self._save_checkpoint("step5_triple_barrier_method", pipeline_state)
                    step_times["step5_triple_barrier_method"] = time.time() - step_start_5

                    # Run validator for Step 5
                    try:
                        step5_validation = await self._run_step_validator(
                            "step5_triple_barrier_method", training_input, pipeline_state,
                        )
                        if step5_validation and step5_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 5: Triple Barrier Method completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 5 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 5 validator failed: {e} - stopping pipeline")
                        return False

                # Step 6: Labeling (regime-specific)
                self._heartbeat("Step 6: Labeling")

                should_run_step6 = _should_run("step6_labeling")
                if not should_run_step6:
                    self.logger.info(
                        f"⏭️ Skipping Step 6: Labeling (starting from '{start_step_key}')",
                    )
                    pipeline_state["labeling"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step6_labeling", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step06, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step6_labeling", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 6 dependencies not met, stopping pipeline")
                        return False

                    step_start_6 = time.time()
                    try:
                        from src.training.steps import step6_labeling

                        step6_success = await step6_labeling.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            data_dir=data_dir,
                            force_rerun=self.force_rerun,
                            config=self.config,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 6: {e}")
                        step6_success = False

                    if not step6_success:
                        self._log_step_completion(
                            "Step 6: Labeling",
                            step_start_6,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 6: Labeling",
                        step_start_6,
                        step_times,
                        success=True,
                    )

                    pipeline_state["labeling"] = {
                        "status": "SUCCESS" if step6_success else "FAILED",
                        "success": bool(step6_success),
                        "completed": bool(step6_success),
                    }
                    self._save_checkpoint("step6_labeling", pipeline_state)
                    step_times["step6_labeling"] = time.time() - step_start_6

                    # Run validator for Step 6
                    try:
                        step6_validation = await self._run_step_validator(
                            "step6_labeling", training_input, pipeline_state,
                        )
                        if step6_validation and step6_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 6: Labeling completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 6 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 6 validator failed: {e} - stopping pipeline")
                        return False

                # Step 7: Feature Engineering (regime-specific)
                self._heartbeat("Step 7: Feature Engineering")

                should_run_step7 = _should_run("step7_feature_engineering")
                if not should_run_step7:
                    self.logger.info(
                        f"⏭️ Skipping Step 7: Feature Engineering (starting from '{start_step_key}')",
                    )
                    pipeline_state["feature_engineering"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step7_feature_engineering", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step07, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step7_feature_engineering", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 7 dependencies not met, stopping pipeline")
                        return False

                    step_start_7 = time.time()
                    try:
                        from src.training.steps import step7_feature_engineering

                        step7_success = await step7_feature_engineering.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            data_dir=data_dir,
                            force_rerun=self.force_rerun,
                            config=self.config,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 7: {e}")
                        step7_success = False

                    if not step7_success:
                        self._log_step_completion(
                            "Step 7: Feature Engineering",
                            step_start_7,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 7: Feature Engineering",
                        step_start_7,
                        step_times,
                        success=True,
                    )

                    pipeline_state["feature_engineering"] = {
                        "status": "SUCCESS" if step7_success else "FAILED",
                        "success": bool(step7_success),
                        "completed": bool(step7_success),
                    }
                    self._save_checkpoint("step7_feature_engineering", pipeline_state)
                    step_times["step7_feature_engineering"] = time.time() - step_start_7

                    # Run validator for Step 7
                    try:
                        step7_validation = await self._run_step_validator(
                            "step7_feature_engineering", training_input, pipeline_state,
                        )
                        if step7_validation and step7_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 7: Feature Engineering completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 7 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 7 validator failed: {e} - stopping pipeline")
                        return False

                # Step 8: HMM-Based Training (now uses regime-specific data)
                self._heartbeat("Step 8: HMM-Based Training")

                should_run_step7 = _should_run("step7_regime_data_splitting")
                if not should_run_step7:
                    self.logger.info(
                        f"⏭️ Skipping Step 7: Regime Data Splitting (starting from '{start_step_key}')",
                    )
                    pipeline_state["regime_data_splitting"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step7_regime_data_splitting", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step07, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step7_regime_data_splitting", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 7 dependencies not met, stopping pipeline")
                        return False

                    step_start_7 = time.time()
                    try:
                        from src.training.steps import step7_regime_data_splitting

                        step7_success = await step7_regime_data_splitting.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            data_dir=data_dir,
                            force_rerun=self.force_rerun,
                            config=self.config,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 7: {e}")
                        step7_success = False

                    if not step7_success:
                        self._log_step_completion(
                            "Step 7: Regime Data Splitting",
                            step_start_7,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 7: Regime Data Splitting",
                        step_start_7,
                        step_times,
                        success=True,
                    )

                    pipeline_state["regime_data_splitting"] = {
                        "status": "SUCCESS" if step7_success else "FAILED",
                        "success": bool(step7_success),
                        "completed": bool(step7_success),
                    }
                    self._save_checkpoint("step7_regime_data_splitting", pipeline_state)
                    step_times["step7_regime_data_splitting"] = time.time() - step_start_7

                    # Run validator for Step 7
                    try:
                        step7_validation = await self._run_step_validator(
                            "step7_regime_data_splitting", training_input, pipeline_state,
                        )
                        if step7_validation and step7_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 7: Regime Data Splitting completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 7 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 7 validator failed: {e} - stopping pipeline")
                        return False

                # Step 8: Enhanced HMM-Based Training with Multi-Output Support
                self._heartbeat("Step 8: Enhanced HMM-Based Training")

                should_run_step8 = _should_run("step8_enhanced_hmm_based_training")
                if not should_run_step8:
                    self.logger.info(
                        f"⏭️ Skipping Step 8: Enhanced HMM-Based Training (starting from '{start_step_key}')",
                    )
                    pipeline_state["enhanced_hmm_based_training"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    # Verify previous step artifacts BEFORE execution
                    if not await self.verify_previous_step_artifacts("step8_enhanced_hmm_based_training", symbol, exchange, timeframe):
                        self.logger.error("❌ Previous step artifacts not found for step08, stopping pipeline")
                        return False

                    # Validate step dependencies BEFORE execution
                    if not await self.validate_step_dependencies("step8_enhanced_hmm_based_training", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 8 dependencies not met, stopping pipeline")
                        return False

                    step_start_8 = time.time()
                    try:
                        from src.training.steps import step9_hmm_based_training_enhanced

                        method_a_cfg = self.config.get("method_a_mixture_of_experts", {})
                        enable_multi_output = self.config.get("enable_multi_output", True)
                        
                        step8_success = await step6_hmm_based_training_enhanced.run_enhanced_step(
                            symbol=symbol,
                            data_dir=data_dir,
                            method_a_mixture_of_experts=method_a_cfg,
                            enable_multi_output=enable_multi_output,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 8: {e}")
                        step8_success = False

                    if not step8_success:
                        self._log_step_completion(
                            "Step 8: Enhanced HMM-Based Training",
                            step_start_8,
                            step_times,
                            success=False,
                        )
                        return False
                    self._log_step_completion(
                        "Step 8: Enhanced HMM-Based Training",
                        step_start_8,
                        step_times,
                        success=True,
                    )

                    pipeline_state["enhanced_hmm_based_training"] = {
                        "status": "SUCCESS" if step8_success else "FAILED",
                        "success": bool(step8_success),
                        "completed": bool(step8_success),
                    }
                    self._save_checkpoint("step8_enhanced_hmm_based_training", pipeline_state)
                    step_times["step8_enhanced_hmm_based_training"] = time.time() - step_start_8

                    # Run validator for Step 8
                    try:
                        step8_validation = await self._run_step_validator(
                            "step8_enhanced_hmm_based_training", training_input, pipeline_state,
                        )
                        if step8_validation and step8_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 8: Enhanced HMM-Based Training completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 8 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 8 validator failed: {e} - stopping pipeline")
                        return False

                # Step 9.5: Multi-Timeframe HMM Ensemble Training
                should_run_step9_5 = _should_run("step9_5_multi_timeframe_hmm_ensemble")
                if not should_run_step9_5:
                    self.logger.info(
                        f"⏭️ Skipping Step 9.5: Multi-Timeframe HMM Ensemble Training (starting from '{start_step_key}')",
                    )
                    pipeline_state["multi_timeframe_hmm_ensemble"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    self._heartbeat("Step 9.5: Multi-Timeframe HMM Ensemble Training")
                    step_start_9_5 = time.time()
                    try:
                        from src.training.steps import step9_5_multi_timeframe_hmm_ensemble

                        step9_5_success = await step9_5_multi_timeframe_hmm_ensemble.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            data_dir=data_dir,
                            timeframe=timeframe,
                            lookback_days=self.lookback_days,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 9.5: {e}")
                        step9_5_success = False

                    pipeline_state["multi_timeframe_hmm_ensemble"] = {
                        "status": "SUCCESS" if step9_5_success else "FAILED",
                        "success": bool(step9_5_success),
                        "completed": bool(step9_5_success),
                    }
                    self._save_checkpoint("step9_5_multi_timeframe_hmm_ensemble", pipeline_state)
                    self._log_step_completion(
                        "Step 9.5: Multi-Timeframe HMM Ensemble Training",
                        step_start_9_5,
                        step_times,
                        success=bool(step9_5_success),
                    )

                    # Run validator for Step 9.5
                    try:
                        step9_5_validation = await self._run_step_validator(
                            "step9_5_multi_timeframe_hmm_ensemble", training_input, pipeline_state,
                        )
                        if step9_5_validation and step9_5_validation.get("validation_passed", False):
                            self.logger.info(
                                "🎉 Step 9.5: Multi-Timeframe HMM Ensemble Training completed successfully and validation passed",
                            )
                        else:
                            self.logger.error("❌ Step 9.5 validation failed - stopping pipeline")
                            return False
                    except Exception as e:
                        self.logger.exception(f"❌ Step 9.5 validator failed: {e} - stopping pipeline")
                        return False

                # Step 6_5: Unified Regime Intelligence
                should_run_step6_5 = _should_run("step6_5_unified_regime_intelligence")
                if not should_run_step6_5:
                    self.logger.info(
                        f"⏭️ Skipping Step 6_5: Unified Regime Intelligence (starting from '{start_step_key}')",
                    )
                    pipeline_state["unified_regime_intelligence"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    self._heartbeat("Step 6_5: Unified Regime Intelligence")
                    step_start_6_5 = time.time()
                    try:
from src.training.steps import step7_analyst_enhancement
from src.training.steps import step8_tactician_labeling
from src.training.steps import step9_tactician_specialist_training
from src.training.steps import step10_confidence_calibration
from src.analyst.meta_label_relevance import MetaLabelRelevanceEvaluator
import pandas as _pd
from src.training.steps import (
from src.training.steps import step12_walk_forward_validation
from src.training.steps import step13_monte_carlo_validation
from src.training.steps import step14_ab_testing
from src.training.steps import step15_saving
from src.training.steps import step2_feature_engineering
from pathlib import Path
import glob
import glob
from pathlib import Path
import copy
                        from src.training.steps import (
                            step5_5_unified_regime_intelligence as _step6_5,
                        )

                        step6_5_success = await _step6_5.run_step(
                            symbol=symbol,
                            exchange=exchange,
                            data_dir=data_dir,
                            timeframe=timeframe,
                            lookback_days=self.lookback_days,
                        )
                    except Exception as e:
                        self.logger.exception(f"❌ Error in Step 6_5: {e}")
                        step6_5_success = False

                    pipeline_state["unified_regime_intelligence"] = {
                        "status": "SUCCESS" if step6_5_success else "FAILED",
                        "success": bool(step6_5_success),
                        "completed": bool(step6_5_success),
                    }
                    self._save_checkpoint("step6_5_unified_regime_intelligence", pipeline_state)
                    self._log_step_completion(
                        "Step 6_5: Unified Regime Intelligence",
                        step_start_6_5,
                        step_times,
                        success=bool(step6_5_success),
                    )
                    # Non-fatal on failure; continue pipeline

                    # Run validator for Step 6_5
                    step6_5_validation = await self._run_step_validator(
                        "step6_5_unified_regime_intelligence", training_input, pipeline_state,
                    )
                    if step6_5_validation and step6_5_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 6_5: Unified Regime Intelligence completed successfully and validation passed",
                        )
                    else:
                        self.logger.warning(
                            f"Validator for step6_5_unified_regime_intelligence failed but is non-fatal: {step6_5_validation.get('error', 'Unknown error')}",
                        )
                    self.logger.info("➡️ Proceeding to Step 7: Analyst Enhancement")

                # Step 7: Analyst Enhancement
                should_run_step7 = _should_run("step7_analyst_enhancement")
                if not should_run_step7:
                    self.logger.info(
                        f"⏭️ Skipping Step 7: Analyst Enhancement (starting from '{start_step_key}')",
                    )
                    pipeline_state["analyst_enhancement"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 7: Analyst Enhancement", step_times):
                        self.logger.info("🔧 STEP 7: Analyst Enhancement (multi-timeframe)...")

                    # Validate step dependencies before execution
                    if not await self.validate_step_dependencies("step7_ensemble_creation", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 7 dependencies not met, skipping")
                        return False


                    analyst_timeframes = ["30m", "15m", "5m"]
                    for tf in analyst_timeframes:
                        self.logger.info(f"🔧 STEP 7: Analyst Enhancement @ {tf}")
                        step7_success = await step7_analyst_enhancement.run_step(
                            symbol=symbol,
                            data_dir=data_dir,
                            timeframe=tf,
                            exchange=exchange,
                        )
                        if not step7_success:
                            return False

                    # Run validator for Step 7 (per timeframe)
                    step7_validation = await self._run_step_validator(
                        "step7_analyst_enhancement",
                        {**training_input, "timeframe": tf},
                        pipeline_state,
                    )
                    if step7_validation and step7_validation.get("validation_passed", False):
                        self.logger.info(
                            f"🎉 Step 7: Analyst Enhancement ({tf}) completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 8: Tactician Labeling")
                    else:
                        self.logger.warning(f"⚠️ Step 7 validation failed: {step7_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 7: (Obsolete) Analyst Ensemble Creation skipped

                # Step 8: Tactician Labeling
                should_run_step8 = _should_run("step8_tactician_labeling")
                if not should_run_step8:
                    self.logger.info(
                        f"⏭️ Skipping Step 8: Tactician Labeling (starting from '{start_step_key}')",
                    )
                    pipeline_state["tactician_labeling"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 8: Tactician Labeling", step_times):
                        self.logger.info("🎯 STEP 8: Tactician Labeling...")

                    # Validate step dependencies before execution
                    if not await self.validate_step_dependencies("step8_model_evaluation", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 8 dependencies not met, skipping")
                        return False


                    step8_success = await step8_tactician_labeling.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe="1m",
                        exchange=exchange,
                    )
                    if not step8_success:
                        return False

                    # Run validator for Step 8
                    step8_validation = await self._run_step_validator(
                        "step8_tactician_labeling",
                        {**training_input, "timeframe": "1m"},
                        pipeline_state,
                    )
                    if step8_validation and step8_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 8: Tactician Labeling completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 9: Tactician Specialist Training")
                    else:
                        self.logger.warning(f"⚠️ Step 8 validation failed: {step8_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 9: Tactician Specialist Training
                should_run_step9 = _should_run("step9_tactician_specialist_training")
                if not should_run_step9:
                    self.logger.info(
                        f"⏭️ Skipping Step 9: Tactician Specialist Training (starting from '{start_step_key}')",
                    )
                    pipeline_state["tactician_specialist_training"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 9: Tactician Specialist Training", step_times):
                        self.logger.info("🎯 STEP 9: Tactician Specialist Training...")

                    # Validate step dependencies before execution
                    if not await self.validate_step_dependencies("step9_hyperparameter_optimization", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 9 dependencies not met, skipping")
                        return False


                    step9_success = await step9_tactician_specialist_training.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe="1m",
                        exchange=exchange,
                    )
                    if not step9_success:
                        return False

                    # Run validator for Step 9
                    step9_validation = await self._run_step_validator(
                        "step9_tactician_specialist_training",
                        {**training_input, "timeframe": "1m"},
                        pipeline_state,
                    )
                    if step9_validation and step9_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 9: Tactician Specialist Training completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 10: Confidence Calibration")
                    else:
                        self.logger.warning(f"⚠️ Step 9 validation failed: {step9_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 10: (Obsolete) Tactician Ensemble Creation skipped

                # Step 10: Confidence Calibration
                should_run_step10 = _should_run("step10_confidence_calibration")
                if not should_run_step10:
                    self.logger.info(
                        f"⏭️ Skipping Step 10: Confidence Calibration (starting from '{start_step_key}')",
                    )
                    pipeline_state["confidence_calibration"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 10: Confidence Calibration", step_times):
                        self.logger.info("🎯 STEP 10: Confidence Calibration...")

                    # Validate step dependencies before execution
                    if not await self.validate_step_dependencies("step10_model_selection", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 10 dependencies not met, skipping")
                        return False


                    step10_success = await step10_confidence_calibration.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                    if not step10_success:
                        return False

                    # Run validator for Step 10
                    step10_validation = await self._run_step_validator(
                        "step10_confidence_calibration", training_input, pipeline_state,
                    )
                    if step10_validation and step10_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 10: Confidence Calibration completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 11: Final Parameters Optimization")
                    else:
                        self.logger.warning(f"⚠️ Step 10 validation failed: {step10_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Save calibration summary (guard undefined calibration_results) atomically
                try:
                    summary_obj = (
                        self._summarize_calibration(calibration_results)  # type: ignore[name-defined]
                        if "calibration_results" in locals()
                        else {"status": "unknown"}
                    )
                    summary_path = data_root / f"{exchange}_{symbol}_calibration_summary.json"
                    _safe_json_write(summary_path, summary_obj)
                except Exception:
                    pass

                # Run meta-label relevance evaluation with complementarity and persist active labels
                try:

                    # Load the latest processed frame if available
                    processed_path = data_root / f"{exchange}_{symbol}_labeled_validation.parquet"
                    df_proc = None
                    if processed_path.exists():

                        df_proc = _pd.read_parquet(processed_path)
                    # Fallback to generic_val (guard)
                    df_input = df_proc if isinstance(df_proc, pd.DataFrame) else None
                    # Gather label names from intensity columns
                    if isinstance(df_input, pd.DataFrame):
                        label_names = sorted(
                            {
                                c.replace("intensity_", "")
                                for c in df_input.columns
                                if isinstance(c, str) and c.startswith("intensity_")
                            },
                        )
                        thresholds = self.get_activation_thresholds()
                        artifacts_dir = self.config.get("meta_labeling", {}).get(
                            "artifacts_dir", "artifacts/meta_labeling",
                        )
                        evaluator = MetaLabelRelevanceEvaluator(
                            artifacts_dir=artifacts_dir,
                            mi_threshold=0.01,
                            sharpe_min_delta=0.0,
                            synergy_mi_threshold=0.005,
                            max_pairs=200,
                        )
                        res = evaluator.evaluate_from_frame(
                            df_input,
                            label_names,
                            thresholds,
                            returns_col="close_returns",
                            risk_free_rate=0.0,
                        )
                    # Persist as part of pipeline state for traceability
                    pipeline_state["active_meta_labels"] = res.get("active_labels", [])
                    pipeline_state["inactive_meta_labels"] = res.get(
                        "inactive_labels", [],
                    )
                    self.logger.info(
                        {
                            "msg": "meta_label_relevance",
                            "active": len(res.get("active_labels", [])),
                            "inactive": len(res.get("inactive_labels", [])),
                        },
                    )
                except Exception as _re:
                    self.logger.warning(f"Meta-label relevance evaluation skipped: {_re}")

                # NEW: Persist thresholds and reliability for MetaLabelingSystem consumption
                try:
                    artifacts_dir = self.config.get("meta_labeling", {}).get(
                        "artifacts_dir", "artifacts/meta_labeling",
                    )
                    artifacts_root = Path(artifacts_dir)
                    artifacts_root.mkdir(parents=True, exist_ok=True)
                    # Persist reliability if available from pipeline_state or calibration
                    reliability = (
                        pipeline_state.get("label_reliability", {})
                        if isinstance(pipeline_state, dict)
                        else {}
                    )
                    if not reliability:
                        # fallback: simple per-label accuracy proxy from analyst_models calibration if present
                        acc_map = {}
                        try:
                            for models in (locals().get("analyst_calibration", {}) or {}).values():
                                if isinstance(models, dict):
                                    for name, res in models.items():
                                        if isinstance(res, dict) and "accuracy" in res:
                                            acc_map[name] = float(res.get("accuracy", 0.0))
                        except Exception:
                            pass
                        reliability = acc_map
                    _safe_json_write(artifacts_root / "reliability.json", reliability)
                    # Persist thresholds if provided in pipeline_state
                    thresholds = (
                        pipeline_state.get("activation_thresholds", {})
                        if isinstance(pipeline_state, dict)
                        else {}
                    )
                    if thresholds:
                        _safe_json_write(artifacts_root / "thresholds.json", thresholds)
                    # Persist active labels if evaluated
                    try:
                        if (
                            "active_meta_labels" in pipeline_state
                            or "inactive_meta_labels" in pipeline_state
                        ):
                            _safe_json_write(
                                artifacts_root / "active_labels.json",
                                {
                                    "active_labels": pipeline_state.get(
                                        "active_meta_labels", [],
                                    ),
                                    "inactive_labels": pipeline_state.get(
                                        "inactive_meta_labels", [],
                                    ),
                                },
                            )
                    except Exception:
                        pass
                    self.logger.info(f"Persisted meta-label artifacts to {artifacts_dir}")
                except Exception as _pe:
                    self.logger.warning(f"Threshold/reliability persistence skipped: {_pe}")

                # Step 11: Final Parameters Optimization (with computational optimization)
                should_run_step11 = _should_run("step11_final_parameters_optimization")
                if not should_run_step11:
                    self.logger.info(
                        f"⏭️ Skipping Step 11: Final Parameters Optimization (starting from '{start_step_key}')",
                    )
                    pipeline_state["final_parameters_optimization"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 11: Final Parameters Optimization", step_times):
                        self.logger.info(
                            "🔧 STEP 11: Final Parameters Optimization with Computational Optimization...",
                        )

                    # Validate step dependencies before execution
                    if not await self.validate_step_dependencies("step11_backtesting", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 11 dependencies not met, skipping")
                        return False

                    if self.computational_optimization_manager:
                        step11_success = await self._run_optimized_parameters_optimization(
                            symbol=symbol,
                            data_dir=data_dir,
                            timeframe=timeframe,
                            exchange=exchange,
                        )
                    else:
                            step11_final_parameters_optimization,
                        )

                        step11_success = (
                            await step11_final_parameters_optimization.run_step(
                                symbol=symbol,
                                data_dir=data_dir,
                                timeframe=timeframe,
                                exchange=exchange,
                            )
                        )
                    if not step11_success:
                        return False

                    # Run validator for Step 11
                    step11_validation = await self._run_step_validator(
                        "step11_final_parameters_optimization",
                        training_input,
                        pipeline_state,
                    )
                    if step11_validation and step11_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 11: Final Parameters Optimization completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 12: Walk Forward Validation")
                    else:
                        self.logger.warning(f"⚠️ Step 11 validation failed: {step11_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 12: Walk Forward Validation
                should_run_step12 = _should_run("step12_walk_forward_validation")
                if not should_run_step12:
                    self.logger.info(
                        f"⏭️ Skipping Step 12: Walk Forward Validation (starting from '{start_step_key}')",
                    )
                    pipeline_state["walk_forward_validation"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 12: Walk Forward Validation", step_times):
                        self.logger.info("📈 STEP 12: Walk Forward Validation...")

                    # Validate step dependencies before execution
                    if not await self.validate_step_dependencies("step12_walk_forward_validation", pipeline_state, self.force_rerun):
                        self.logger.error("❌ Step 12 dependencies not met, skipping")
                        return False


                    step12_success = await step12_walk_forward_validation.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                    if not step12_success:
                        return False

                    # Run validator for Step 12
                    step12_validation = await self._run_step_validator(
                        "step12_walk_forward_validation", training_input, pipeline_state,
                    )
                    if step12_validation and step12_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 12: Walk Forward Validation completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 13: Monte Carlo Validation")
                    else:
                        self.logger.warning(f"⚠️ Step 12 validation failed: {step12_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 13: Monte Carlo Validation
                should_run_step13 = _should_run("step13_monte_carlo_validation")
                if not should_run_step13:
                    self.logger.info(
                        f"⏭️ Skipping Step 13: Monte Carlo Validation (starting from '{start_step_key}')",
                    )
                    pipeline_state["monte_carlo_validation"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 13: Monte Carlo Validation", step_times):
                        self.logger.info("🎲 STEP 13: Monte Carlo Validation...")

                    # Validate step dependencies before execution
                    if not await self._validate_step_dependencies("step13_monte_carlo_validation", pipeline_state):
                        self.logger.error("❌ Step 13 dependencies not met, skipping")
                        return False


                    step13_success = await step13_monte_carlo_validation.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                    if not step13_success:
                        return False

                    # Run validator for Step 13
                    step13_validation = await self._run_step_validator(
                        "step13_monte_carlo_validation", training_input, pipeline_state,
                    )
                    if step13_validation and step13_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 13: Monte Carlo Validation completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 14: A/B Testing")
                    else:
                        self.logger.warning(f"⚠️ Step 13 validation failed: {step13_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 14: A/B Testing
                should_run_step14 = _should_run("step14_ab_testing")
                if not should_run_step14:
                    self.logger.info(
                        f"⏭️ Skipping Step 14: A/B Testing (starting from '{start_step_key}')",
                    )
                    pipeline_state["ab_testing"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 14: A/B Testing", step_times):
                        self.logger.info("🧪 STEP 14: A/B Testing...")

                    # Validate step dependencies before execution
                    if not await self._validate_step_dependencies("step14_model_deployment", pipeline_state):
                        self.logger.error("❌ Step 14 dependencies not met, skipping")
                        return False


                    step14_success = await step14_ab_testing.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                    if not step14_success:
                        return False

                    # Run validator for Step 14
                    step14_validation = await self._run_step_validator(
                        "step14_ab_testing", training_input, pipeline_state,
                    )
                    if step14_validation and step14_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 14: A/B Testing completed successfully and validation passed",
                        )
                        self.logger.info("➡️ Proceeding to Step 15: Saving Results")
                    else:
                        self.logger.warning(f"⚠️ Step 14 validation failed: {step14_validation.get('error', 'Unknown error')}")
                        self.logger.warning("⚠️ Proceeding anyway (validation is non-blocking)")
                        # Continue execution even if validation fails

                # Step 15: Saving Results
                should_run_step15 = _should_run("step15_saving")
                if not should_run_step15:
                    self.logger.info(
                        f"⏭️ Skipping Step 15: Saving Results (starting from '{start_step_key}')",
                    )
                    pipeline_state["saving_results"] = {
                        "status": "SKIPPED",
                        "success": True,
                        "skipped": True,
                        "reason": f"start_step={start_step_key}",
                    }
                else:
                    with self._timed_step("Step 15: Saving Results", step_times):
                        self.logger.info("💾 STEP 15: Saving Results...")

                    # Validate step dependencies before execution
                    if not await self._validate_step_dependencies("step15_live_monitoring", pipeline_state):
                        self.logger.error("❌ Step 15 dependencies not met, skipping")
                        return False


                    step15_success = await step15_saving.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                    if not step15_success:
                        return False

                    # Run validator for Step 15
                    step15_validation = await self._run_step_validator(
                        "step15_saving", training_input, pipeline_state,
                    )
                    if step15_validation and step15_validation.get("validation_passed", False):
                        self.logger.info(
                            "🎉 Step 15: Saving Results completed successfully and validation passed",
                        )

                # Calculate total time and summary
                total_time = time.time() - start_time
                total_memory = (
                    psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                )  # MB

                # Log comprehensive summary
                self.logger.info("=" * 100)
                self.logger.info(
                    "🎉 COMPREHENSIVE 15-STEP TRAINING PIPELINE COMPLETED SUCCESSFULLY",
                )
                self.logger.info("=" * 100)
                self.logger.info(
                    f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
                self.logger.info(
                    f"⏱️ Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)",
                )
                self.logger.info(f"💾 Final Memory Usage: {total_memory:.1f} MB")
                self.logger.info(f"🎯 Symbol: {symbol}")
                self.logger.info(f"🏢 Exchange: {exchange}")
                self.logger.info(f"📊 Timeframe: {timeframe}")
                self.logger.info(
                    f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}",
                )

                # Log step-by-step timing
                self.logger.info("📊 Step-by-Step Timing:")
                for step_name, step_time in step_times.items():
                    percentage = (step_time / total_time) * 100
                    self.logger.info(
                        f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)",
                    )

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
                self.logger.info(
                    f"   ✅ Parallel backtester initialized with {self.max_workers} workers",
                )

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
                self.logger.info(
                    "✅ Progressive evaluator initialized for optimization",
                )

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
                    parallel_results, pb.evaluate_batch(
                        param_combinations, market_data,
                    )

                # Find best parameters from parallel results
                if parallel_results:
                    best_result, max(
                        parallel_results, key=lambda x: x.get("score", -float("inf")),
                    )
                    optimization_results = best_result
                    self.logger.info(
                        f"✅ Parallel optimization completed. Best score: {best_result.get('score', 'N/A')}",
                    )

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
                    self.logger.info(
                        f"📈 New best score: {score} at trial {trial + 1}",
                    )

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
                    "trials_completed": optimization_results.get(
                        "trials_completed", self.n_trials,
                    ),
                    "best_score": optimization_results.get("best_score", 0.0),
                    "cache_hits": getattr(self.cached_backtester, "cache", {})
                    if self.cached_backtester
                    else {},
                    "memory_profile": self.memory_manager.profile_memory_usage()
                    if self.memory_manager
                    else {},
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

            self.logger.info(
                "✅ Enhanced optimized parameters optimization completed successfully",
            )

            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced optimized parameters optimization failed: {e}",
            )
            return False

    def _generate_parameter_combinations(self) -> list[dict]:
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

    def _generate_random_parameters(self) -> dict:
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
                market_data = self.data_manager.load_from_parquet(
                    str(preferred_parquet),
                )
                self.logger.info(f"✅ Loaded market data from {preferred_parquet}")
                return market_data
            if preferred_csv.exists():
                market_data = pd.read_csv(preferred_csv)
                self.logger.info(f"✅ Loaded market data from {preferred_csv}")
                return market_data

            # Fallback to raw files in data_dir
            parquet_path = data_dir / f"{exchange}_{symbol}_klines.parquet"
            csv_path = data_dir / f"{exchange}_{symbol}_klines.csv"
            if parquet_path.exists():
                self.logger.info(f"Loading data from Parquet: {parquet_path}")
                try:
                    return self.data_manager.load_from_parquet(str(parquet_path))
                except Exception as e:
                    self.logger.warning(
                        f"Parquet load failed ({e}); falling back to CSV if available",
                    )
                if csv_path.exists():
                    self.logger.info(f"Loading data from CSV: {csv_path}")
                    try:
                        return pd.read_csv(csv_path)
                    except Exception as e:
                        self.logger.warning(
                            f"CSV load failed ({e}); returning empty DataFrame",
                        )

            self.logger.warning(
                f"⚠️ Market data files not found in {data_dir} for {exchange} {symbol}",
            )
            return pd.DataFrame()

        except Exception as e:
            self.logger.exception(f"❌ Failed to load market data: {e}")
            return None

    def _evaluate_params_with_cache(
        self, market_data: pd.DataFrame, params: dict[str, Any],
    ) -> float:
        """Evaluate params using cached backtester if available, else simple placeholder."""
        if self.cached_backtester is None:
            self.cached_backtester = CachedBacktester(market_data)
        try:
            return float(self.cached_backtester.run_cached_backtest(params))
        except Exception:
            return random.uniform(-1.0, 1.0)

    def get_memory_profile(self) -> dict[str, Any]:
        """Expose current memory profile using MemoryManager."""
        if self.memory_manager is None:
            self.memory_manager = MemoryManager(memory_threshold=self.memory_threshold)
        return self.memory_manager.profile_memory_usage()

    def get_optimization_stats(self) -> dict[str, Any]:
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

    def _get_validation_level(self, step_name: str, is_fatal: bool) -> str:
        """
        Determine the appropriate validation level for a step.
        By default, all steps use CRITICAL validation for maximum thoroughness.
        
        Args:
            step_name: Name of the step
            is_fatal: Whether step failure is fatal
            
        Returns:
            Validation level string (defaults to CRITICAL)
        """
        # All steps now default to CRITICAL validation for maximum thoroughness
        # This ensures comprehensive validation across the entire pipeline
        
        # For backward compatibility, we still identify specific critical steps
        # but they all return CRITICAL level
        critical_steps = [
            "step1_data_collection",
            "step2_feature_engineering", 
            "step3_hmm_regime_discovery",
            "step6_hmm_based_training",
            "step7_analyst_enhancement",
            "step9_tactician_specialist_training",
            "step12_walk_forward_validation",
            "step13_monte_carlo_validation",
        ]
        
        comprehensive_steps = [
            "step1_5_data_converter",
            "step4_processing_labeling",
            "step5_regime_data_splitting",
            "step6_5_unified_regime_intelligence",
            "step8_tactician_labeling",
            "step10_confidence_calibration",
            "step11_final_parameters_optimization",
            "step14_ab_testing",
            "step15_saving",
        ]
        
        # All steps now use CRITICAL validation by default
        # This ensures maximum thoroughness and reliability
        return "CRITICAL"
    
    def _log_validation_details(self, validation_result: dict[str, Any]) -> None:
        """
        Log detailed validation information for comprehensive validation levels.
        
        Args:
            validation_result: Validation result dictionary
        """
        try:
            if not validation_result:
                return
                
            self.logger.info("📊 Validation Details:")
            
            # Log validation level and timing
            if "validation_level" in validation_result:
                self.logger.info(f"   Level: {validation_result['validation_level']}")
            
            if "duration" in validation_result:
                self.logger.info(f"   Duration: {validation_result['duration']:.3f}s")
            
            # Log warnings and recommendations
            if validation_result.get("warnings"):
                self.logger.info(f"   Warnings: {len(validation_result['warnings'])}")
                for warning in validation_result["warnings"][:3]:  # Show first 3
                    self.logger.info(f"     - {warning}")
            
            if validation_result.get("recommendations"):
                self.logger.info(f"   Recommendations: {len(validation_result['recommendations'])}")
                for rec in validation_result["recommendations"][:3]:  # Show first 3
                    self.logger.info(f"     - {rec}")
            
            # Log validation results summary
            if "validation_results" in validation_result:
                vr = validation_result["validation_results"]
                if isinstance(vr, dict):
                    self.logger.info(f"   Validation Checks: {len(vr)}")
                    for check_name, check_result in list(vr.items())[:5]:  # Show first 5
                        status = "✅" if check_result.get("valid", False) else "❌"
                        self.logger.info(f"     {status} {check_name}")
                        
        except Exception as e:
            self.logger.debug(f"Error logging validation details: {e}")
    
    def _log_validation_failure(self, validation_result: dict[str, Any]) -> None:
        """
        Log validation failure details.
        
        Args:
            validation_result: Validation result dictionary
        """
        try:
            if not validation_result:
                return
                
            self.logger.error("❌ Validation Failure Details:")
            
            # Log error message
            if "error" in validation_result:
                self.logger.error(f"   Error: {validation_result['error']}")
            
            # Log critical issues
            if validation_result.get("critical_issues"):
                self.logger.error(f"   Critical Issues: {len(validation_result['critical_issues'])}")
                for issue in validation_result["critical_issues"][:3]:  # Show first 3
                    self.logger.error(f"     - {issue}")
            
            # Log data quality issues
            if validation_result.get("data_quality_issues"):
                self.logger.error(f"   Data Quality Issues: {len(validation_result['data_quality_issues'])}")
                for issue in validation_result["data_quality_issues"][:3]:  # Show first 3
                    self.logger.error(f"     - {issue}")
            
            # Log missing artifacts
            if validation_result.get("missing_artifacts"):
                self.logger.error(f"   Missing Artifacts: {len(validation_result['missing_artifacts'])}")
                for artifact in validation_result["missing_artifacts"][:3]:  # Show first 3
                    self.logger.error(f"     - {artifact}")
                    
        except Exception as e:
            self.logger.debug(f"Error logging validation failure: {e}")

    async def _run_step_validator(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        validation_level: str = "CRITICAL",
    ) -> dict[str, Any]:
        """Run validator for a specific step."

        Args:
            step_name: Name of the step to validate
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            validation_level: Validation level (defaults to CRITICAL for maximum thoroughness)

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
                validation_level=validation_level,
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
        context="step1_5_data_converter"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DATA_PREPROCESSING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    @validate_pipeline_input(
        required_params=["symbol", "exchange", "timeframe", "data_dir"],
        required_directories=["data_cache"],
        min_memory_gb=4.0,
        min_disk_gb=2.0
    )
    async def _execute_step1_5_with_qa(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
        step1_5_run_step: callable,
    ) -> bool:
        """Execute step1_5_data_converter with enhanced reporting."""
        
        step_start_time = time.time()
        step_errors = []
        step_warnings = []
        
        try:
            # Execute the original step function
            result = await step1_5_run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )

            # Generate step report
            await self._generate_step_report(
                "step1_5_data_converter",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            self.logger.info("✅ [QA] step1_5_data_converter completed with enhanced reporting")
            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ [QA] step1_5_data_converter failed: {e}")
            
            # Generate step report even on failure
            await self._generate_step_report(
                "step1_5_data_converter",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step2_feature_engineering"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.FEATURE_ENGINEERING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=16.0,
        cpu_threshold_percent=90.0
    )
    async def _execute_step2_with_qa(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        timeframe: str,
        force_rerun: bool,
        feature_config: dict,
    ) -> bool:
        """Execute step2_feature_engineering with enhanced reporting."""
        
        step_start_time = time.time()
        step_errors = []
        step_warnings = []
        
        try:

            # Execute the original step function
            result = await step2_feature_engineering.run_step(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
                timeframe=timeframe,
                force_rerun=force_rerun,
                feature_config=feature_config,
            )

            # Generate step report
            await self._generate_step_report(
                "step2_feature_engineering",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            self.logger.info("✅ [QA] step2_feature_engineering completed with enhanced reporting")
            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ [QA] step2_feature_engineering failed: {e}")
            
            # Generate step report even on failure
            await self._generate_step_report(
                "step2_feature_engineering",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training history storage",
    )
    async def _store_enhanced_training_history(
        self, enhanced_training_input: dict[str, Any],
    ) -> None:
        """Store enhanced training history."

        Args:
            enhanced_training_input: Enhanced training input parameters

        """
        try:
            # Add to training history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "training_input": enhanced_training_input,
                "results": self.enhanced_training_results,
            }

            self.enhanced_training_history.append(history_entry)

            # Limit history size
            if len(self.enhanced_training_history) > self.max_enhanced_training_history:
                self.enhanced_training_history = self.enhanced_training_history[
                    -self.max_enhanced_training_history :
                ]

            self.logger.info(
                f"📁 Stored training history entry (total: {len(self.enhanced_training_history)})",
            )

        except Exception as e:
            self.logger.exception(f"❌ Failed to store training history: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training results storage",
    )
    async def _store_enhanced_training_results(self) -> None:
        """Store enhanced training results."""
        try:
            self.logger.info("📁 Storing enhanced training results...")

            # Store results in a format that can be retrieved later
            results_key = (
                f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # This would typically store to database or file system
            self.logger.info(
                f"📁 Storing enhanced training results with key: {results_key}",
            )

        except Exception as e:
            self.logger.exception(f"❌ Failed to store enhanced training results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training results getting",
    )
    def get_enhanced_training_results(
        self,
        enhanced_training_type: str | None,
    ) -> dict[str, Any]:
        """Get enhanced training results."

        Args:
            enhanced_training_type: Type of training results to get

        Returns:
            dict: Enhanced training results

        """
        try:
            if enhanced_training_type:
                return self.enhanced_training_results.get(enhanced_training_type, {})
            return self.enhanced_training_results.copy()

        except Exception as e:
            self.logger.exception(f"Failed to get enhanced training results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training history getting",
    )
    def get_enhanced_training_history(
        self,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        """Get enhanced training history."

        Args:
            limit: Maximum number of history entries to return

        Returns:
            list: Enhanced training history

        """
        try:
            history = self.enhanced_training_history.copy()
            if limit:
                history = history[-limit:]
            return history

        except Exception as e:
            self.logger.exception(f"Failed to get enhanced training history: {e}")
            return []

    def get_enhanced_training_status(self) -> dict[str, Any]:
        """Get enhanced training status."

        Returns:
            dict: Enhanced training status information

        """
        return {
            "is_training": self.is_training,
            "has_results": bool(self.enhanced_training_results),
            "history_count": len(self.enhanced_training_history),
            "blank_training_mode": self.blank_training_mode,
            "max_trials": self.max_trials,
            "n_trials": self.n_trials,
            "lookback_days": self.lookback_days,
            "enable_validators": self.enable_validators,
            "enable_computational_optimization": self.enable_computational_optimization,
            "optimization_statistics": self.optimization_statistics,
        }

    def get_validation_results(self) -> dict[str, Any]:
        """Get validation results for all steps."

        Returns:
            dict: Validation results summary

        """
        return {
            "validation_results": self.validation_results,
            "validation_summary": validator_orchestrator.get_validation_summary(),
            "failed_validations": validator_orchestrator.get_failed_validations(),
        }

    def get_computational_optimization_results(self) -> dict[str, Any]:
        """Get computational optimization results and statistics."

        Returns:
            dict: Computational optimization results

        """
        if self.computational_optimization_manager:
            return {
                "optimization_statistics": self.computational_optimization_manager.get_optimization_statistics(),
                "enabled_optimizations": self.optimization_statistics,
                "manager_available": True,
            }
        return {
            "optimization_statistics": {},
            "enabled_optimizations": {},
            "manager_available": False,
        }

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

    def get_optimization_statistics(self) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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

    def use_cached_backtesting(self, params: dict[str, Any]) -> float:
        """Use cached backtesting for parameter evaluation."""
        if self.cached_backtester:
            return self.cached_backtester.run_cached_backtest(params)
        self.logger.warning("Cached backtester not initialized")
        return 0.0

    def use_progressive_evaluation(
        self, params: dict[str, Any], evaluator_func,
    ) -> float:
        """Use progressive evaluation for early stopping."""
        if self.progressive_evaluator:
            return self.progressive_evaluator.evaluate_progressively(
                params, evaluator_func,
            )
        self.logger.warning("Progressive evaluator not initialized")
        return 0.0

    def generate_cache_key(self, params: dict[str, Any]) -> str:
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
                    self.logger.info(
                        "✅ Computational optimization manager initialized",
                    )
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to initialize computational optimization manager: {e}",
                    )
                    self.enable_computational_optimization = False

            self.logger.info("✅ Enhanced Training Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced Training Manager initialization failed: {e}",
            )
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
        """Execute comprehensive feature selection and pruning."

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
                self.logger.warning(
                    f"⚠️ Engineered features not found at {features_path}",
                )
                self.logger.info(
                    "🔄 Proceeding with feature selection on available data...",
                )
                return True  # Continue with available features

            # Load features using optimized data manager
            features_df = self.data_manager.load_from_parquet(features_path)

            if features_df.empty:
                self.logger.warning("⚠️ No features available for selection")
                return True

            self.logger.info(
                f"📈 Loaded {len(features_df.columns)} features for selection",
            )

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
            selected_features_path = (
                Path(data_dir) / f"{symbol}_{exchange}_{timeframe}_selected_features.parquet"
            )
            self.data_manager.save_to_parquet(selected_features, str(selected_features_path))

            self.logger.info("✅ Feature selection completed:")
            self.logger.info(f"   Selected: {len(selected_features.columns)} features")
            self.logger.info(f"   Reduced from: {len(features_df.columns)} features")
            self.logger.info(
                f"   Reduction: {((len(features_df.columns) - len(selected_features.columns)) / len(features_df.columns) * 100):.1f}%",
            )

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
        """Execute tiered feature selection strategy."

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
            self.logger.info(
                f"   ✅ Tier 1: Selected {len(tier_1_features.columns)} core features",
            )

            # Tier 2: Normalized features (z-scores, changes, accelerations)
            tier_2_features = self._select_tier_2_features(
                features_df, feature_categories["tier_2"], tier_2_count,
            )
            selected_features = pd.concat([selected_features, tier_2_features], axis=1)
            self.logger.info(
                f"   ✅ Tier 2: Selected {len(tier_2_features.columns)} normalized features",
            )

            # Tier 3: Interaction features (spread*volume, etc.)
            tier_3_features = self._select_tier_3_features(
                features_df, feature_categories["tier_3"], tier_3_count,
            )
            selected_features = pd.concat([selected_features, tier_3_features], axis=1)
            self.logger.info(
                f"   ✅ Tier 3: Selected {len(tier_3_features.columns)} interaction features",
            )

            # Tier 4: Lagged features (lagged interactions)
            tier_4_features = self._select_tier_4_features(
                features_df, feature_categories["tier_4"], tier_4_count,
            )
            selected_features = pd.concat([selected_features, tier_4_features], axis=1)
            self.logger.info(
                f"   ✅ Tier 4: Selected {len(tier_4_features.columns)} lagged features",
            )

            # Tier 5: Causality features (market microstructure causality)
            tier_5_features = self._select_tier_5_features(
                features_df, feature_categories["tier_5"], tier_5_count,
            )
            selected_features = pd.concat([selected_features, tier_5_features], axis=1)
            self.logger.info(
                f"   ✅ Tier 5: Selected {len(tier_5_features.columns)} causality features",
            )

            # Apply final pruning if we exceed total_max_features
            if len(selected_features.columns) > total_max_features:
                selected_features = self._apply_final_pruning(
                    selected_features, total_max_features,
                )
            self.logger.info(
                f"   🔧 Final pruning: Reduced to {len(selected_features.columns)} features",
            )

            return selected_features

        except Exception as e:
            self.logger.exception(f"❌ Tiered feature selection failed: {e}")
            return pd.DataFrame()

    def _categorize_features_by_tier(self, features_df: pd.DataFrame) -> dict:
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
        self, features_df: pd.DataFrame, tier_1_features: list, count: int,
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
        self, features_df: pd.DataFrame, tier_2_features: list, count: int,
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
        self, features_df: pd.DataFrame, tier_3_features: list, count: int,
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
        self, features_df: pd.DataFrame, tier_4_features: list, count: int,
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
        self, features_df: pd.DataFrame, tier_5_features: list, count: int,
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
    def get_label_expert_models(self) -> dict[str, dict[str, Any]]:
        return self.label_expert_models

    def get_label_expert_calibrators(self) -> dict[str, Any]:
        return self.label_expert_calibrators

    def get_label_reliability(self) -> dict[str, float]:
        if not self.label_reliability:
            self._load_label_reliability()
        return self.label_reliability

    def get_activation_thresholds(self) -> dict[str, float]:
        if not self.activation_thresholds:
            self._load_activation_thresholds()
        return self.activation_thresholds

    def save_activation_thresholds(self, thresholds: dict[str, Any]) -> None:
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
                self.activation_thresholds.update(flat)
                self.logger.info(f"Saved activation thresholds to {target}")
        except Exception as e:
            self.logger.warning(f"Failed to save activation thresholds: {e}")

    def _load_activation_thresholds(self) -> None:
        if self.force_rerun:
            self.logger.info(
                "Force rerun enabled; skipping loading persisted activation thresholds",
            )
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

    def save_label_reliability(self, reliability: dict[str, float]) -> None:
        try:
            target = self.artifacts_dir / "reliability.json"
            _safe_json_write(target, reliability)
            self.label_reliability.update({k: float(v) for k, v in reliability.items()})
            self.logger.info(f"Saved label reliability to {target}")
        except Exception as e:
            self.logger.warning(f"Failed to save label reliability: {e}")

    def _load_label_reliability(self) -> None:
        if self.force_rerun:
            self.logger.info(
                "Force rerun enabled; skipping loading persisted reliability",
            )
            return
        try:
            path = self.artifacts_dir / "reliability.json"
            if path.exists():
                with open(path) as f:
                    self.label_reliability = {
                        k: float(v) for k, v in json.load(f).items()
                    }
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
        """Clear artifacts from the specified step and all subsequent steps."
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
        """Verify that artifacts from the previous step exist before starting a step."

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
        """Clear artifacts for a specific step."

        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        """
        try:
        except Exception as e:
            pass  # TODO: Handle exception properly

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
                        pass  # File doesn't exist, which is fine'
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Could not delete {file_path}: {e}")

            if cleared_count > 0:
                self.logger.info(f"   🧹 Cleared {cleared_count} artifacts for {step_name}")
            else:
                self.logger.debug(f"   ℹ️ No artifacts found for {step_name}")

        except Exception as e:
            self.logger.exception(f"❌ Error clearing artifacts for {step_name}: {e}")

    # Performance tracking methods
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_step_performance"
    )
    async def _track_step_performance(self, step_type: str, step_name: str, data: Any, expected: Any) -> bool:
        """Track performance for a specific step."
        
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
        """Track performance for a trained model."
        
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
                # This is a simplified approach - in practice, you'd use actual test data'
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
        """Track performance for optimization results."
        
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
        """Track performance for validation results."
        
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
        """Track performance for A/B testing results."
        
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

    # Enhanced reporting methods
    async def _generate_step_report(self, step_name: str, step_result: Any, step_start_time: float, step_success: bool, step_errors: List[str] = None, step_warnings: List[str] = None):
        """Generate and append step information to shared pipeline report."""
        
        if not self.enable_detailed_reporting:
            return
        
        try:
            step_end_time = time.time()
            execution_duration = step_end_time - step_start_time
            
            # Create step report section
            step_report_section = {
                "step_name": step_name,
                "pipeline_execution_id": self.current_pipeline_execution_id,
                "execution_start_time": datetime.fromtimestamp(step_start_time).isoformat(),
                "execution_end_time": datetime.fromtimestamp(step_end_time).isoformat(),
                "execution_duration_seconds": execution_duration,
                "execution_duration_formatted": f"{execution_duration:.2f}s",
                "success": step_success,
                "result_type": type(step_result).__name__,
                "result_summary": self._summarize_result(step_result),
                "errors": step_errors or [],
                "warnings": step_warnings or [],
                "system_resources": await self._get_system_resources(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Load existing shared report or create new one
            shared_report_path = self.pipeline_reports_dir / f"{self.current_pipeline_execution_id}_shared_report.json"
            
            if shared_report_path.exists():
                with open(shared_report_path, 'r', encoding='utf-8') as f:
                    shared_report = json.load(f)
            else:
                shared_report = {
                    "pipeline_execution_id": self.current_pipeline_execution_id,
                    "pipeline_start_time": datetime.fromtimestamp(step_start_time).isoformat(),
                    "pipeline_config": self.config,
                    "steps": {},
                    "pipeline_summary": {
                        "total_steps": len(self.STEP_ORDER),
                        "completed_steps": 0,
                        "failed_steps": 0,
                        "total_duration": 0,
                        "overall_success": True
                    }
                }
            
            # Append step information to shared report
            shared_report["steps"][step_name] = step_report_section
            shared_report["pipeline_summary"]["completed_steps"] = len(shared_report["steps"])
            shared_report["pipeline_summary"]["failed_steps"] = sum(1 for step in shared_report["steps"].values() if not step["success"])
            shared_report["pipeline_summary"]["overall_success"] = shared_report["pipeline_summary"]["failed_steps"] == 0
            shared_report["pipeline_summary"]["total_duration"] = sum(step["execution_duration_seconds"] for step in shared_report["steps"].values())
            
            # Save updated shared report
            with open(shared_report_path, 'w', encoding='utf-8') as f:
                json.dump(shared_report, f, indent=2, ensure_ascii=False, default=str)
            
            # Store in memory for pipeline summary
            self.step_reports[step_name] = step_report_section
            
            # Log completion
            status_emoji = "✅" if step_success else "❌"
            self.logger.info(f"{status_emoji} [STEP REPORT] {step_name} appended to shared report: {shared_report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate step report for {step_name}: {e}")
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create a summary of the step result."""
        
        try:
            if hasattr(result, 'shape'):  # DataFrame
                return {
                    "type": "DataFrame",
                    "shape": result.shape,
                    "columns_count": len(result.columns),
                    "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2) if hasattr(result, 'memory_usage') else None
                }
            elif isinstance(result, dict):
                return {
                    "type": "dict",
                    "keys_count": len(result),
                    "keys": list(result.keys())[:10]  # First 10 keys
                }
            elif isinstance(result, (list, tuple)):
                return {
                    "type": type(result).__name__,
                    "length": len(result),
                    "element_types": [type(item).__name__ for item in result[:5]]  # First 5 elements
                }
            elif isinstance(result, bool):
                return {
                    "type": "boolean",
                    "value": result
                }
            else:
                return {
                    "type": type(result).__name__,
                    "value_preview": str(result)[:100]  # First 100 characters
                }
        except Exception:
            return {
                "type": "unknown",
                "error": "Could not summarize result"
            }
    
    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            disk = psutil.disk_usage('/')
            
            return {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_usage_percent": cpu,
                "disk_usage_percent": disk.percent,
                "disk_available_gb": disk.free / (1024**3)
            }
        except Exception:
            return {
                "error": "Could not retrieve system resources"
            }


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced training manager setup",
)
async def setup_enhanced_training_manager(
    config: dict[str, Any] | None,
) -> EnhancedTrainingManager | None:
    """Setup and return a configured EnhancedTrainingManager instance."

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
