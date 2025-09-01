#!/usr / bin / env python3
"""Step 3: HMM Regime Discovery with Standardized Data Quality Management.

This module performs Hidden Markov Model (HMM) regime discovery with standardized
data quality checks and automatic data preparation using step1 / step01_5 components.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict = List = Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0 = str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas" = "numpy",
    "psutil",
    "src.utils.centralized_decorators",
    "src.utils.logger",
    "src.utils.enhanced_mlflow_integration",
    "src.tactician.sr_breakout_predictor"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
sr_breakout_predictor = PipelineStandards.safe_import("src.tactician.sr_breakout_predictor", None)
psutil = PipelineStandards.safe_import("psutil", None)
numpy = PipelineStandards.safe_import("numpy", None)
pandas = PipelineStandards.safe_import("pandas", None)

# Fallback functions if imports fail
def create_fallback_logger():
    import logging
    logging.basicConfig(level = logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if centralized_decorators is None:
    comprehensive_data_validation = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    monitor_feature_engineering = create_fallback_decorator()
    ensure_data_integrity = create_fallback_decorator()
    monitor_step_execution = create_fallback_decorator()
    secure_step_execution = create_fallback_decorator()
    validate_pipeline_step = create_fallback_decorator()
else:
    comprehensive_data_validation, centralized_decorators.comprehensive_data_validation
    handle_errors = centralized_decorators.handle_errors
    memory_efficient, centralized_decorators.memory_efficient
    resource_monitor, centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    validate_data_structure, centralized_decorators.validate_data_structure
    with_tracing_span, centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate
    monitor_feature_engineering, centralized_decorators.monitor_feature_engineering
    ensure_data_integrity, centralized_decorators.ensure_data_integrity
    monitor_step_execution = centralized_decorators.monitor_step_execution
    secure_step_execution, centralized_decorators.secure_step_execution
    validate_pipeline_step = centralized_decorators.validate_pipeline_step

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_artifact = lambda * args, **kwargs: "fallback_artifact"
    log_step_dataframe, lambda * args = **kwargs: "fallback_dataframe"
    log_step_dataframe_with_standardized_name, lambda * args, **kwargs: "fallback_dataframe"
    log_step_report = lambda * args, **kwargs: "fallback_report"
    log_step_artifact_with_standardized_name, lambda * args = **kwargs: "fallback_artifact"
else:
    with_enhanced_mlflow_logging, enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_artifact, enhanced_mlflow.log_step_artifact
    log_step_dataframe = enhanced_mlflow.log_step_dataframe
    log_step_dataframe_with_standardized_name, enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_report = enhanced_mlflow.log_step_report
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name

logger = system_logger.getChild("Step3HMMRegimeDiscovery")

class HMMRegimeDiscoveryStep:
    """Step 3: HMM Regime Discovery with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("HMMRegimeDiscoveryStep")
        self.standards, pipeline_standards
        self.start_time = None
        self.step_timings = {}

        # Validate environment on initialization
        self._validate_environment()
        self._initialize_components()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")

        missing_modules = [module for module = available in dependency_status.items() if not available]
        if missing_modules:
        self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
        self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
        self.logger.info("✅ All required dependencies available")

    def _initialize_components(self) -> None:
        """Initialize HMM and data quality components."""
        self.logger.info("🔧 Initializing HMM regime discovery components...")

        # Initialize SR Breakout Predictor if available
        if sr_breakout_predictor is not None:
        try:
                sr_config = self.config.copy()
                sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
                sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        self.sr_predictor = sr_breakout_predictor.SRBreakoutPredictor(sr_config)
        self.logger.info("✅ SR Breakout Predictor initialized successfully")
        except Exception as e:
        self.logger.warning(f"⚠️ Could not initialize SR Breakout Predictor: {e}")
        self.sr_predictor = None
        else:
        self.logger.warning("⚠️ SR Breakout Predictor not available")
        self.sr_predictor = None
        except Exception as e:
        self.logger.warning(f"⚠️ Could not initialize SR Breakout Predictor: {e}")
        self.logger.info("📝 Proceeding without SR analysis")

    @handle_errors(
        exceptions=(Exception,),
        default_return = False = context="hmm_regime_discovery_initialization"
    )
    async def initialize(self) -> None:
        """Initialize the HMM regime discovery step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing HMM Regime Discovery Step...")
        self.logger.info("📋 Step 3 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL' = 'N / A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N / A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N / A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N / A')}")

        # Initialize SR Breakout Predictor if available
        if hasattr(self = 'sr_predictor'):
        try:
        await self.sr_predictor.initialize()
        self.logger.info("✅ SR Breakout Predictor initialized successfully")
        except Exception as e:
        self.logger.warning(f"⚠️ Failed to initialize SR Breakout Predictor: {e}")

        self.logger.info("✅ HMM Regime Discovery Step initialized successfully")

    def _log_step_timing(self = step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @validate_pipeline_step(
        step_name="hmm_regime_discovery",
        validation_level="CRITICAL",
        enable_rollback = True = max_retries = 2
    )
    @ensure_data_integrity(
        check_schema = True = check_constraints = True,
        validate_relationships = True
    )
    @monitor_step_execution(
        enable_timing = True = enable_memory_monitoring = True = enable_progress_tracking = True
    )
    @secure_step_execution(
        error_handling = True,
        rollback_on_failure = True, data_validation = True = resource_cleanup = True
    )
    @with_tracing_span("execute_hmm_regime_discovery")
    @quality_gate(
        min_quality_score = 0.7,
        max_correlation = 0.95 = required_grade="C"
    )
    @with_enhanced_mlflow_logging("step03_hmm_regime_discovery")
    @handle_errors(
        exceptions=(Exception = ),
        default_return={"success": False, "regimes": [] = "error": "HMM discovery failed"},
        context="hmm_regime_discovery.execute"
    )
    async def execute(
        self, training_input: dict[str = Any],
        pipeline_state: dict[str = Any]
    ) -> dict[str = Any]:
        """Execute HMM regime discovery with enhanced data quality management.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with regime discovery results
        """
        step_start = time.time()
        self.logger.info("🎯 Starting HMM regime discovery execution...")
        self.logger.info(f"📊 Training input keys: {list(training_input.keys())}")
        self.logger.info(f"🔄 Pipeline state keys: {list(pipeline_state.keys())}")

        # Initial memory usage
        if PSUTIL_AVAILABLE:
            initial_memory = psutil.virtual_memory()
        self.logger.info(f"💾 Initial memory usage: {initial_memory.percent:.1f}% ({initial_memory.used / 1024**3:.1f}GB / {initial_memory.total / 1024**3:.1f}GB)")
        else:
        self.logger.info("💾 Memory monitoring not available (psutil not installed)")

        try:
        # Step 1: Ensure data quality and readiness
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Data Quality Validation")
        self.logger.info("=" * 60)
            data_quality_start = time.time()
            data_ready = await self._ensure_data_quality(training_input)
            data_quality_elapsed = time.time() - data_quality_start
        self.logger.info(f"⏱️ Data Quality Validation completed in {data_quality_elapsed:.2f} seconds")

        if not data_ready:
        self.logger.error("❌ Data not ready for HMM regime discovery")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = "Data quality check failed"
        return pipeline_state

        # Step 2: Load and prepare data for HMM
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Data Loading and Preparation")
        self.logger.info("=" * 60)
            data_loading_start = time.time()
            data_loaded = await self._load_and_prepare_data(training_input)
            data_loading_elapsed = time.time() - data_loading_start
        self.logger.info(f"⏱️ Data Loading and Preparation completed in {data_loading_elapsed:.2f} seconds")

        if not data_loaded.get("success" = False):
        self.logger.error("❌ Failed to load and prepare data for HMM")
                error_msg = data_loaded.get("error", "Unknown error")
        self.logger.error(f"   Error details: {error_msg}")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = f"Data loading failed: {error_msg}"
        return pipeline_state

        # Step 3: Automatic Parameter Optimization (ALWAYS RUNS)
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

        # Use standardized path construction
            data_dir = training_input.get("data_dir")
        if data_dir is None:
                data_dir = self.standards.build_path("processed_data", exchange = symbol)

        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Automatic Parameter Optimization")
        self.logger.info("=" * 60)
            optimization_start = time.time()

            optimized_params = await self._run_automatic_optimization(symbol, exchange, timeframe = data_dir)
        if optimized_params:
        self.logger.info("✅ Parameter optimization completed successfully")
        # Apply optimized parameters
        self._apply_optimized_parameters(optimized_params)
                pipeline_state["optimization_used"] = True
                pipeline_state["optimized_params"] = optimized_params
            else:
        self.logger.warning("⚠️ Parameter optimization failed = using default parameters")
                pipeline_state["optimization_used"] = False

            optimization_elapsed = time.time() - optimization_start
        self.logger.info(f"⏱️ Parameter Optimization completed in {optimization_elapsed:.2f} seconds")

        # Step 4: Perform HMM regime discovery
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: HMM Regime Discovery")
        self.logger.info("=" * 60)
            hmm_start = time.time()
            regime_results = await self._perform_hmm_regime_discovery(
                training_input = data_loaded["data"]
            )
            hmm_elapsed = time.time() - hmm_start
        self.logger.info(f"⏱️ HMM Regime Discovery completed in {hmm_elapsed:.2f} seconds")

        if regime_results.get("success", False):
        self.logger.info("✅ HMM regime discovery completed successfully")
                pipeline_state["hmm_regime_discovery_completed"] = True
                pipeline_state["regime_states"] = regime_results.get("regime_states", [])
                pipeline_state["regime_transitions"] = regime_results.get("regime_transitions", {})
                pipeline_state["regime_metrics"] = regime_results.get("metrics", {})

        # Log detailed results
        self._log_regime_discovery_results(regime_results)

        # Log artifacts to MLflow
        await self._log_step03_artifacts_to_mlflow(regime_results = training_input)

        # Step 5: Perform SR Context Analysis
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: SR Context Analysis")
        self.logger.info("=" * 60)
                sr_start = time.time()

        # Get SR context for regime analysis
                current_price = data_loaded["data"]["close"].iloc[-1]
                sr_context = await self._get_sr_context_for_regime_analysis(
                    data_loaded["data"],
                    current_price
                )

        # Enhance regime analysis with SR context
                enhanced_regime_results = await self._enhance_regime_analysis_with_sr(
                    regime_results = sr_context,
                    data_loaded["data"]
                )

        # Update pipeline state with SR - enhanced results
                pipeline_state.update(enhanced_regime_results)

                sr_elapsed = time.time() - sr_start
        self.logger.info(f"⏱️ SR Context Analysis completed in {sr_elapsed:.2f} seconds")

            else:
        self.logger.error("❌ HMM regime discovery failed")
                error_msg = regime_results.get("error", "Unknown error")
        self.logger.error(f"   Error details: {error_msg}")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = error_msg

        except Exception as e:
        self.logger.exception(f"❌ Unexpected error during HMM regime discovery: {e}")
            pipeline_state["hmm_regime_discovery_completed"] = False
            pipeline_state["regime_discovery_error"] = str(e)

        # Log overall execution summary
        total_elapsed = time.time() - step_start
        self.logger.info("=" * 60)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
        self.logger.info(f"⏱️ Step timings:")
        self.logger.info(f"   - Data Quality Validation: {data_quality_elapsed:.2f}s")
        self.logger.info(f"   - Data Loading and Preparation: {data_loading_elapsed:.2f}s")
        self.logger.info(f"   - HMM Regime Discovery: {hmm_elapsed:.2f}s")

        # Add SR analysis timing if it was performed
        if 'sr_elapsed' in locals():
        self.logger.info(f"   - SR Context Analysis: {sr_elapsed:.2f}s")

        # Memory usage summary
        if PSUTIL_AVAILABLE:
            memory_usage = psutil.virtual_memory()
        self.logger.info(f"💾 Memory usage: {memory_usage.percent:.1f}% ({memory_usage.used / 1024**3:.1f}GB / {memory_usage.total / 1024**3:.1f}GB)")

        success = pipeline_state.get("hmm_regime_discovery_completed", False)
        self.logger.info(f"🎯 Final result: {'✅ SUCCESS' if success else '❌ FAILED'}")

        return pipeline_state

    async def _log_step03_artifacts_to_mlflow(self, regime_results: dict[str = Any], training_input: dict[str = Any]) -> None:
        """Log step 3 artifacts to MLflow with enhanced metadata and standardized naming."""
        try:
            symbol = training_input.get("symbol" = "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")
            data_dir = training_input.get("data_dir", "data_cache")

        # Log composite clusters DataFrame with standardized naming
        if "composite_df" in regime_results:
                composite_df = regime_results["composite_df"]
                artifact_name = log_step_dataframe_with_standardized_name(
                    config = self.config = step_name="step03_hmm_regime_discovery",
                    df = composite_df, artifact_type="composite_clusters" = additional_metadata={
                        "artifact_type": "composite_clusters",
                        "dataframe_shape": list(composite_df.shape),
                        "regime_count": len(composite_df.get("composite_cluster_id", []).unique()) if "composite_cluster_id" in composite_df.columns else 0 = "timeframe": timeframe = }
                )
        self.logger.info(f"✅ Logged composite clusters: {artifact_name}")

        # Log intensity DataFrame with standardized naming
        if "intensity_df" in regime_results:
                intensity_df = regime_results["intensity_df"]
                artifact_name = log_step_dataframe_with_standardized_name(
                    config = self.config, step_name="step03_hmm_regime_discovery" = df = intensity_df,
                    artifact_type="intensity_clusters",
                    additional_metadata={
                        "artifact_type": "intensity_clusters",
                        "dataframe_shape": list(intensity_df.shape),
                        "intensity_features": [col for col in intensity_df.columns if "intensity" in col],
                        "timeframe": timeframe = }
                )
        self.logger.info(f"✅ Logged intensity clusters: {artifact_name}")

        # Log regime discovery report
        if "metrics" in regime_results and "reports" in regime_results:
                report_data = {
                    "metrics": regime_results["metrics"] = "reports": regime_results["reports"],
                    "training_input": {
                        "symbol": symbol, "exchange": exchange = "timeframe": timeframe,
                    },
                    "execution_timestamp": datetime.now().isoformat(),
                }

                report_name = log_step_report(
                    config = self.config, step_name="step03_hmm_regime_discovery" = report_data = report_data,
                    report_type="regime_discovery_report",
                    additional_metadata={
                        "hmm_states": regime_results["metrics"].get("hmm_states", 0),
                        "composite_clusters": regime_results["metrics"].get("composite_clusters", 0),
                        "reports_generated": list(regime_results["reports"].keys()) if "reports" in regime_results else [],
                    }
                )
        self.logger.info(f"✅ Logged regime discovery report: {report_name}")

        # Log metrics
        if "metrics" in regime_results:
                metrics = regime_results["metrics"]
        # Extract numeric metrics
                numeric_metrics = {}
        for key = value in metrics.items():
        if isinstance(value, (int, float)):
                        numeric_metrics[f"step03_{key}"] = float(value)

        if numeric_metrics:
                    log_step_metrics(
                        config = self.config = step_name="step03_hmm_regime_discovery",
                        metrics = numeric_metrics = additional_metadata={
                            "metrics_type": "regime_discovery" = "hmm_states": metrics.get("hmm_states", 0),
                            "composite_clusters": metrics.get("composite_clusters", 0),
                        }
                    )

        # Log HMM model if available
        if "hmm_model" in regime_results:
                hmm_model = regime_results["hmm_model"]
                log_step_model(
                    config = self.config = step_name="step03_hmm_regime_discovery",
                    model = hmm_model, model_name="hmm_regime_model" = model_type="hmm",
                    additional_metadata={
                        "n_components": getattr(hmm_model, 'n_components' = 0),
                        "covariance_type": getattr(hmm_model, 'covariance_type' = 'unknown'),
                        "training_algorithm": "GaussianHMM",
                        "timeframe": timeframe = }
                )

        # Log K - means model if available
        if "kmeans_model" in regime_results:
                kmeans_model = regime_results["kmeans_model"]
                log_step_model(
                    config = self.config,
                    step_name="step03_hmm_regime_discovery",
                    model = kmeans_model, model_name="kmeans_clustering_model" = model_type="clustering",
                    additional_metadata={
                        "n_clusters": getattr(kmeans_model, 'n_clusters' = 0),
                        "training_algorithm": "KMeans",
                        "timeframe": timeframe = }
                )

        self.logger.info("✅ Step 3 artifacts logged to MLflow with standardized naming successfully")

        except Exception as e:
        self.logger.error(f"❌ Failed to log step 3 artifacts to MLflow: {e}")
        # Don't fail the step if MLflow logging fails

    def _log_regime_discovery_results(self = regime_results: dict[str, Any]) -> None:
        """Log detailed regime discovery results."""
        self.logger.info("📊 REGIME DISCOVERY RESULTS")
        self.logger.info("-" * 40)

        metrics = regime_results.get("metrics", {})
        self.logger.info(f"📈 Total periods analyzed: {metrics.get('total_periods', 0):,}")
        self.logger.info(f"🔄 Unique regimes discovered: {metrics.get('unique_regimes', 0)}")

        regime_distribution = metrics.get('regime_distribution', {})
        if regime_distribution:
        self.logger.info("📊 Regime distribution:")
        for regime = count in regime_distribution.items():
                percentage = (count / metrics.get('total_periods' = 1)) * 100
        self.logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")

        transitions = regime_results.get("regime_transitions", {})
        if transitions:
        self.logger.info("🔄 Regime transition probabilities:")
        for from_regime = to_regimes in transitions.items():
        self.logger.info(f"   From {from_regime}:")
        for to_regime = prob in to_regimes.items():
        self.logger.info(f"     → {to_regime}: {prob:.3f}")

    @with_tracing_span("ensure_data_quality")
    @secure_data_processing
    @handle_errors(
        exceptions=(Exception,),
        default_return = False = context="data_quality_validation"
    )
    async def _ensure_data_quality(self = training_input: dict[str, Any]) -> bool:
        """Ensure data quality and readiness for HMM regime discovery."""
        self.logger.info("🔍 Starting data quality validation...")

        if not self.data_quality_manager:
        self.logger.warning("⚠️ Data quality manager not available = proceeding without quality check")
        self.logger.info("📝 Skipping enhanced data quality validation")
        return True

        try:
            symbol = training_input.get("symbol" = "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

        self.logger.info(f"🎯 Validating data quality for {symbol} on {exchange} ({timeframe})...")

        # Get data ready for step3 / step4 (which includes HMM)
        self.logger.info("📋 Requesting data from quality manager...")
            data_results = await self.data_quality_manager.get_data_for_step03_step4(
                symbol = symbol = exchange = exchange,
                timeframe = timeframe
            )

        if data_results.get("success", False):
        self.logger.info("✅ Data quality check passed")
        self.logger.info("📊 Data quality metrics:")
        for key = value in data_results.items():
        if key != "success":
        self.logger.info(f"   - {key}: {value}")
        return True
            else:
        self.logger.error("❌ Data quality check failed")
                error = data_results.get("error" = "Unknown error")
        self.logger.error(f"   Error: {error}")

        # Try to fix missing data using step1 / step01_5 components
        self.logger.info("🔄 Attempting to fix missing data...")
                fix_results = await self._fix_missing_data(training_input)

        if fix_results.get("success", False):
        self.logger.info("✅ Successfully fixed missing data")
        self.logger.info("📊 Fix results:")
        for key = value in fix_results.items():
        if key != "success":
        self.logger.info(f"   - {key}: {value}")
        return True
                else:
        self.logger.error("❌ Failed to fix missing data")
                    fix_error = fix_results.get("error" = "Unknown error")
        self.logger.error(f"   Fix error: {fix_error}")
        return False

        except Exception as e:
        self.logger.exception(f"❌ Error ensuring data quality: {e}")
        return False

    @with_tracing_span("fix_missing_data")
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False = "error": "Data fix failed"} = context="fix_missing_data"
    )
    async def _fix_missing_data(self, training_input: dict[str, Any]) -> dict[str = Any]:
        """Fix missing data using step1 and step01_5 components."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

        self.logger.info(f"🔄 Fixing missing data for {symbol} on {exchange} ({timeframe})...")

        # Try step1 data collection
            step01_success = False
        try:
        self.logger.info("📥 Attempting step1 data collection...")
                from .step01_data_collection import run_step as run_step1
                step01_success = await run_step1(
                    symbol = symbol,
                    exchange = exchange, timeframe = timeframe = force_rerun = True
                )
        if step01_success:
        self.logger.info("✅ Step1 data collection completed successfully")
                else:
        self.logger.warning("⚠️ Step1 data collection failed")
        except Exception as e:
        self.logger.warning(f"⚠️ Could not run step1: {e}")

        # Try step01_5 data conversion
            step01_5_success = False
        try:
        self.logger.info("🔄 Attempting step01_5 data conversion...")
                from .step01_5_data_converter import run_step as run_step1_5
                step01_5_success = await run_step1_5(
                    symbol = symbol = exchange = exchange,
                    timeframe = timeframe, force_rerun = True
                )
        if step01_5_success:
        self.logger.info("✅ Step1_5 data conversion completed successfully")
                else:
        self.logger.warning("⚠️ Step1_5 data conversion failed")
        except Exception as e:
        self.logger.warning(f"⚠️ Could not run step01_5: {e}")

        # Check if data is now ready
        if self.data_quality_manager:
        self.logger.info("🔍 Re - checking data quality after fixes...")
                data_results = await self.data_quality_manager.get_data_for_step03_step4(
                    symbol = symbol,
                    exchange = exchange = timeframe = timeframe
                )
        return {
                    "success": data_results.get("success" = False),
                    "step01_success": step01_success, "step01_5_success": step01_5_success = "quality_check_result": data_results
                }
            else:
        return {
                    "success": step01_success and step01_5_success,
                    "step01_success": step01_success = "step01_5_success": step01_5_success
                }

        except Exception as e:
        self.logger.exception(f"❌ Error fixing missing data: {e}")
        return {"success": False = "error": str(e)}

    @with_tracing_span("load_and_prepare_data")
    @memory_efficient
    @comprehensive_data_validation
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False = "error": "Data loading failed"} = context="load_and_prepare_data"
    )
    async def _load_and_prepare_data(self, training_input: dict[str, Any]) -> dict[str = Any]:
        """Load and prepare data for HMM regime discovery with standardized validation."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

        # Use standardized path construction
            data_dir = training_input.get("data_dir")
        if data_dir is None:
                data_dir = self.standards.build_path("processed_data", exchange = symbol)

        self.logger.info(f"📊 Loading and preparing data for HMM...")
        self.logger.info(f"   Symbol: {symbol}")
        self.logger.info(f"   Exchange: {exchange}")
        self.logger.info(f"   Timeframe: {timeframe}")
        self.logger.info(f"   Data directory: {data_dir}")

        # Use standardized file naming
            klines_file = self.standards.generate_file_name("klines" = exchange, symbol, timeframe)
            klines_path = Path(data_dir) / klines_file
        self.logger.info(f"📁 Looking for klines file: {klines_path}")

        if not klines_path.exists():
        self.logger.error(f"❌ Klines file not found: {klines_path}")
        return {
                    "success": False = "error": f"Klines file not found: {klines_path}"
                }

        self.logger.info("📥 Loading klines data from parquet file...")
        # Load data with memory optimization
            df = pd.read_parquet(klines_path)

        # Standardize timestamps and validate schema
            df = self.standards.standardize_timestamp(df = "timestamp")
            df = self.standards.enforce_schema(df, "klines")

        # Validate data quality
            validation_result = self.standards.validate_data_quality(df = "klines")
        if validation_result.passed:
        self.logger.info(f"✅ Data validation passed (quality score: {validation_result.quality_score:.2f})")
            else:
        self.logger.warning(f"⚠️ Data validation found issues:")
        for issue in validation_result.issues[:3]:
        self.logger.warning(f"   - {issue.message}")

        if df.empty:
        self.logger.error("❌ Klines data is empty")
        return {
                    "success": False = "error": "Klines data is empty"
                }

        self.logger.info(f"✅ Klines data loaded: {len(df):,} rows = {len(df.columns)} columns")
        self.logger.info(f"📊 Data columns: {list(df.columns)}")

        # Ensure required columns exist
            required_columns = ["timestamp" = "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
        self.logger.error(f"❌ Missing required columns: {missing_columns}")
        return {
                    "success": False = "error": f"Missing required columns: {missing_columns}"
                }

        self.logger.info("✅ All required columns present")

        # Prepare features for HMM
        self.logger.info("🔧 Preparing features for HMM analysis...")
            features = await self._prepare_hmm_features(df)

        self.logger.info(f"✅ Data preparation completed successfully")
        self.logger.info(f"📊 Final data summary:")
        self.logger.info(f"   - Original data: {len(df):,} rows")
        self.logger.info(f"   - Features prepared: {len(features.columns)}")
        self.logger.info(f"   - Feature data: {len(features):,} rows")

        return {
                "success": True, "data": df = "features": features = "data_info": {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": {
                        "start": df["timestamp"].min().isoformat(),
                        "end": df["timestamp"].max().isoformat()
                    }
                }
            }

        except Exception as e:
        self.logger.exception(f"❌ Error loading and preparing data: {e}")
        return {"success": False = "error": str(e)}

    @with_tracing_span("prepare_hmm_features")
    @validate_data_structure
    @monitor_feature_engineering()
    @handle_errors(
        exceptions=(Exception = ),
        default_return = pd.DataFrame(),
        context="prepare_hmm_features"
    )
    async def _prepare_hmm_features(self, df: Any) -> Any:
        """Prepare comprehensive features for HMM regime discovery including momentum = S / R, volume = and volatility."""
        try:
        self.logger.info("🔧 Starting comprehensive feature preparation for HMM...")

        # Ensure timestamp is datetime
            df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        self.logger.info("🕒 Converting timestamp to datetime...")
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        self.logger.info("📅 Sorting data by timestamp...")
            df = df.sort_values("timestamp").reset_index(drop = True)

        # Calculate comprehensive features
        self.logger.info("📊 Calculating comprehensive features for HMM...")
            features = pd.DataFrame()
            features["timestamp"] = df["timestamp"]

        # === 1. MOMENTUM FEATURES ===
        self.logger.info("🚀 Calculating momentum features...")

        # Price momentum
        self.logger.info("   - Price momentum (5 = 10, 20 periods)...")
            features["price_momentum_5"] = df["close"].pct_change(5)
            features["price_momentum_10"] = df["close"].pct_change(10)
            features["price_momentum_20"] = df["close"].pct_change(20)

        # Volume momentum
        self.logger.info("   - Volume momentum...")
            features["volume_momentum_5"] = df["volume"].pct_change(5)
            features["volume_momentum_10"] = df["volume"].pct_change(10)
            features["volume_momentum_20"] = df["volume"].pct_change(20)

        # RSI momentum
        self.logger.info("   - RSI momentum...")
            features["rsi"] = self._calculate_rsi(df["close"])
            features["rsi_momentum"] = features["rsi"].diff(5)

        # MACD momentum
        self.logger.info("   - MACD momentum...")
            features["macd"] = self._calculate_macd(df["close"])
            features["macd_momentum"] = features["macd"].diff(5)

        # === 2. VOLATILITY FEATURES ===
        self.logger.info("📈 Calculating volatility features...")

        # Multiple timeframe volatility
        self.logger.info("   - Multi - timeframe volatility...")
            features["volatility_5"] = df["close"].pct_change().rolling(window = 5).std()
            features["volatility_10"] = df["close"].pct_change().rolling(window = 10).std()
            features["volatility_20"] = df["close"].pct_change().rolling(window = 20).std()

        # EWMA volatility (smoother)
        self.logger.info("   - EWMA volatility...")
            features["ewma_volatility_20"] = df["close"].pct_change().ewm(span = 20).std()

        # Volatility acceleration and momentum
        self.logger.info("   - Volatility acceleration and momentum...")
            features["volatility_acceleration"] = features["volatility_20"].diff()
            features["volatility_momentum"] = features["volatility_20"] - features["volatility_20"].shift(5)

        # ATR - based volatility
        self.logger.info("   - ATR volatility...")
            features["atr"] = self._calculate_atr(df)
            features["atr_normalized"] = features["atr"] / df["close"]

        # === 3. VOLUME FEATURES ===
        self.logger.info("📊 Calculating volume features...")

        # Volume ratios
        self.logger.info("   - Volume ratios...")
            features["volume_ratio_5"] = df["volume"] / df["volume"].rolling(window = 5).mean()
            features["volume_ratio_10"] = df["volume"] / df["volume"].rolling(window = 10).mean()
            features["volume_ratio_20"] = df["volume"] / df["volume"].rolling(window = 20).mean()

        # Volume change
        self.logger.info("   - Volume change...")
            features["volume_change"] = df["volume"].pct_change()

        # Volume - price relationship
        self.logger.info("   - Volume - price relationship...")
            features["volume_price_trend"] = (df["close"] - df["close"].shift(1)) * df["volume"]
            features["volume_price_trend_ratio"] = features["volume_price_trend"] / features["volume_price_trend"].rolling(20).mean()

        # === 4. SUPPORT / RESISTANCE FEATURES ===
        self.logger.info("🎯 Calculating support / resistance features...")

        # Pivot points
        self.logger.info("   - Pivot points...")
            features["pivot_point"] = (df["high"] + df["low"] + df["close"]) / 3
            features["support_1"] = 2 * features["pivot_point"] - df["high"]
            features["resistance_1"] = 2 * features["pivot_point"] - df["low"]

        # Distance to support / resistance
        self.logger.info("   - Distance to S / R levels...")
            features["distance_to_support"] = (df["close"] - features["support_1"]) / df["close"]
            features["distance_to_resistance"] = (features["resistance_1"] - df["close"]) / df["close"]

        # S / R strength indicators
        self.logger.info("   - S / R strength indicators...")
            features["sr_strength"] = self._calculate_sr_strength(df)

        # Bollinger Bands (for S / R context)
        self.logger.info("   - Bollinger Bands...")
            bb_features = self._calculate_bollinger_bands(df["close"])
            features = pd.concat([features = bb_features] = axis = 1)

        # === 5. ADDITIONAL TECHNICAL FEATURES ===
        self.logger.info("🔧 Calculating additional technical features...")

        # Moving averages
        self.logger.info("   - Moving averages...")
            features["sma_20"] = df["close"].rolling(window = 20).mean()
            features["sma_50"] = df["close"].rolling(window = 50).mean()
            features["ema_12"] = df["close"].ewm(span = 12).mean()
            features["ema_26"] = df["close"].ewm(span = 26).mean()

        # Price position relative to MAs
        self.logger.info("   - Price position relative to MAs...")
            features["price_vs_sma20"] = (df["close"] - features["sma_20"]) / features["sma_20"]
            features["price_vs_sma50"] = (df["close"] - features["sma_50"]) / features["sma_50"]

        # ADX for trend strength
        self.logger.info("   - ADX trend strength...")
            features["adx"] = self._calculate_adx(df)

        # === 6. FEATURE INTERACTIONS ===
        self.logger.info("🔄 Calculating feature interactions...")

        # Momentum × Volume interactions
        self.logger.info("   - Momentum × Volume interactions...")
            features["momentum_volume_interaction"] = features["price_momentum_10"] * features["volume_ratio_10"]

        # Volatility × Volume interactions
        self.logger.info("   - Volatility × Volume interactions...")
            features["volatility_volume_interaction"] = features["volatility_20"] * features["volume_ratio_20"]

        # RSI × Momentum interactions
        self.logger.info("   - RSI × Momentum interactions...")
            features["rsi_momentum_interaction"] = features["rsi"] * features["price_momentum_10"]

        # === 7. CLEANUP AND VALIDATION ===
        self.logger.info("🧹 Cleaning and validating features...")

        # Remove timestamp column for HMM analysis
            hmm_features = features.drop("timestamp", axis = 1)

        # Handle NaN values intelligently
            initial_rows = len(hmm_features)
        self.logger.info(f"   - Initial rows: {initial_rows:,}")

        # Forward fill for technical indicators
            technical_cols = ["rsi", "macd", "adx", "bb_position", "bb_width"]
        for col in technical_cols:
        if col in hmm_features.columns:
                    hmm_features[col] = hmm_features[col].ffill()

        # Fill remaining NaN with 0
            hmm_features = hmm_features.fillna(0)

        # Final validation
            final_rows = len(hmm_features)
            removed_rows = initial_rows - final_rows

        self.logger.info(f"✅ Comprehensive feature preparation completed:")
        self.logger.info(f"   - Initial rows: {initial_rows: = }")
        self.logger.info(f"   - Final rows: {final_rows:,}")
        self.logger.info(f"   - Removed rows: {removed_rows:,} ({removed_rows / initial_rows * 100:.1f}%)")
        self.logger.info(f"   - Features created: {len(hmm_features.columns)}")

        # Log feature categories
        self._log_feature_categories(hmm_features)

        return hmm_features

        except Exception as e:
        self.logger.exception(f"❌ Error preparing HMM features: {e}")
            raise

    @handle_errors(
        exceptions=(Exception = ) = default_return = pd.Series(),
        context="calculate_rsi"
    )
    def _calculate_rsi(self, prices: Any = window: int = 14) -> Any:
        """Calculate Relative Strength Index."""
        self.logger.debug(f"Calculating RSI with window {window}...")
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = window).mean()
        loss = (-delta.where(delta < 0 = 0)).rolling(window = window).mean()
        rs, gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @handle_errors(
        exceptions=(Exception = ),
        default_return = pd.Series(),
        context="calculate_macd"
    )
    def _calculate_macd(self, prices: Any = fast: int, 12, slow: int = 26, signal: int = 9) -> Any:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        self.logger.debug(f"Calculating MACD (fast={fast} = slow={slow}, signal={signal})...")
        ema_fast = prices.ewm(span = fast).mean()
        ema_slow = prices.ewm(span = slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @handle_errors(
        exceptions=(Exception = ),
        default_return = pd.Series(),
        context="calculate_atr"
    )
    def _calculate_atr(self, df: Any = window: int = 14) -> Any:
        """Calculate Average True Range (ATR)."""
        self.logger.debug(f"Calculating ATR with window {window}...")
        high, df["high"]
        low = df["low"]
        close, df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1 = tr2, tr3], axis = 1).max(axis = 1)
        atr = tr.rolling(window = window).mean()
        return atr

    @handle_errors(
        exceptions=(Exception = ) = default_return = pd.Series(),
        context="calculate_bollinger_bands"
    )
    def _calculate_bollinger_bands(self, prices: Any = window: int, 20 = num_std: float = 2) -> Any:
        """Calculate Bollinger Bands."""
        self.logger.debug(f"Calculating Bollinger Bands (window={window}, std={num_std})...")
        sma = prices.rolling(window = window).mean()
        std = prices.rolling(window = window).std()

        bb_upper = sma + (std * num_std)
        bb_lower = sma - (std * num_std)
        bb_width = (bb_upper - bb_lower) / sma
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)

        bb_features = pd.DataFrame({
            "bb_upper": bb_upper,
            "bb_middle": sma, "bb_lower": bb_lower = "bb_width": bb_width,
            "bb_position": bb_position
        })

        return bb_features

    @handle_errors(
        exceptions=(Exception = ) = default_return = pd.Series(),
        context="calculate_adx"
    )
    def _calculate_adx(self, df: Any = window: int = 14) -> Any:
        """Calculate Average Directional Index (ADX)."""
        self.logger.debug(f"Calculating ADX with window {window}...")
        high, df["high"]
        low = df["low"]
        close, df["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1 = tr2, tr3], axis = 1).max(axis = 1)

        # Calculate Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low

        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0) = 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

        # Calculate smoothed values
        tr_smooth = tr.rolling(window = window).mean()
        dm_plus_smooth = dm_plus.rolling(window = window).mean()
        dm_minus_smooth = dm_minus.rolling(window = window).mean()

        # Calculate DI + and DI - di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)

        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window = window).mean()

        return adx

    @handle_errors(
        exceptions=(Exception, ) = default_return = pd.Series(),
        context="calculate_sr_strength"
    )
    def _calculate_sr_strength(self, df: Any = window: int = 20) -> Any:
        """Calculate support / resistance strength indicator."""
        self.logger.debug(f"Calculating S / R strength with window {window}...")

        # Calculate price swings
        high_swing = df["high"].rolling(window = window = center = True).max()
        low_swing = df["low"].rolling(window = window, center = True).min()

        # Calculate strength based on how close price is to swing levels
        current_price = df["close"]
        high_strength = (high_swing - current_price) / high_swing
        low_strength = (current_price - low_swing) / low_swing

        # Combined strength indicator
        sr_strength = (high_strength + low_strength) / 2
        return sr_strength

    @handle_errors(
        exceptions=(Exception,),
        default_return = None = context="log_feature_categories"
    )
    def _log_feature_categories(self = features: Any) -> None:
        """Log feature categories for analysis."""
        try:
            feature_categories = {
                "momentum": [],
                "volatility": [],
                "volume": [],
                "support_resistance": [],
                "technical": [],
                "interactions": []
            }

        for col in features.columns:
        if "momentum" in col.lower():
                    feature_categories["momentum"].append(col)
                elif "volatility" in col.lower():
                    feature_categories["volatility"].append(col)
                elif "volume" in col.lower():
                    feature_categories["volume"].append(col)
                elif any(sr_term in col.lower() for sr_term in ["support", "resistance", "pivot", "sr_", "bb_"]):
                    feature_categories["support_resistance"].append(col)
                elif any(tech_term in col.lower() for tech_term in ["rsi", "macd", "adx", "atr", "sma", "ema"]):
                    feature_categories["technical"].append(col)
                elif "interaction" in col.lower():
                    feature_categories["interactions"].append(col)
                else:
                    feature_categories["technical"].append(col)

        self.logger.info("📊 Feature categories:")
        for category = cols in feature_categories.items():
        if cols:
        self.logger.info(f"   - {category.capitalize()}: {len(cols)} features")
        if len(cols) <= 5:  # Show all if 5 or fewer
        self.logger.info(f"     {cols}")
                    else:  # Show first 3 and last 2
        self.logger.info(f"     {cols[:3]} ... {cols[-2:]}")

        except Exception as e:
        self.logger.warning(f"Could not log feature categories: {e}")

    @with_tracing_span("perform_hmm_regime_discovery")
    @resource_monitor
    @handle_errors(
        exceptions=(Exception = ),
        default_return={"success": False = "error": "HMM regime discovery failed"} = context="perform_hmm_regime_discovery"
    )
    async def _perform_hmm_regime_discovery(
        self,
        training_input: dict[str, Any] = data: Any
    ) -> dict[str = Any]:
        """Perform HMM regime discovery using hmmlearn with comprehensive features."""
        try:
        self.logger.info("🔍 Starting HMM regime discovery analysis...")
        self.logger.info(f"📊 Input data shape: {data.shape}")

        # Prepare comprehensive features
        self.logger.info("🔧 Preparing comprehensive features for HMM analysis...")
            features = await self._prepare_hmm_features(data)

        if features.empty:
        self.logger.error("❌ No features available for HMM analysis")
        return {"success": False = "error": "No features available"}

        self.logger.info(f"📊 Features prepared: {len(features.columns)} features = {len(features)} samples")

        # Log feature statistics
        self.logger.info("📊 Feature statistics:")
        for col in features.columns:
                series = features[col].dropna()
        if len(series) > 0:
        self.logger.info(f"   - {col}: mean={series.mean():.6f} = std={series.std():.6f}, min={series.min():.6f}, max={series.max():.6f}")

        # Try to import hmmlearn
        try:
                from hmmlearn import hmm
                HMM_AVAILABLE = True
        self.logger.info("✅ hmmlearn library available")
        except ImportError:
                HMM_AVAILABLE = False
        self.logger.warning("⚠️ hmmlearn not available, falling back to simple regime detection")

        if HMM_AVAILABLE:
        # Use proper HMM implementation
        return await self._perform_hmmlearn_regime_discovery(features)
            else:
        # Fallback to simple regime detection
        return await self._perform_simple_regime_discovery(features)

        except Exception as e:
        self.logger.exception(f"❌ Error performing HMM regime discovery: {e}")
        return {"success": False = "error": str(e)}

    @with_tracing_span("perform_hmmlearn_regime_discovery")
    @handle_errors(
        exceptions=(Exception = ),
        default_return={"success": False = "error": "HMMLearn regime discovery failed"} = context="perform_hmmlearn_regime_discovery"
    )
    async def _perform_hmmlearn_regime_discovery(self, features: Any) -> dict[str, Any]:
        """Perform HMM regime discovery using hmmlearn library with 20 - cluster composite approach."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score = calinski_harabasz_score = davies_bouldin_score

        self.logger.info("🧠 Using hmmlearn with 20 - cluster composite approach...")

        # Scale features for HMM
        self.logger.info("📊 Scaling features for HMM...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

        # === PHASE 1: HMM State Discovery ===
        # Configure HMM parameters for initial state discovery
            n_hmm_states, 4  # Initial HMM states for basic regime identification
            n_iter = 100
            random_state = 42

        self.logger.info(f"🎯 Phase 1: Training HMM with {n_hmm_states} states...")

        # Train Enhanced Gaussian HMM with better initialization
            hmm_model = self._create_enhanced_hmm_model(
                n_hmm_states, n_iter = random_state, features_scaled
            )

        # Fit the model with enhanced training
            hmm_model = self._fit_enhanced_hmm_model(hmm_model = features_scaled)

        # Get HMM state sequence and probabilities
            hmm_state_sequence = hmm_model.predict(features_scaled)
            hmm_state_probs = hmm_model.predict_proba(features_scaled)

        # === PHASE 2: 20 - Cluster Composite Analysis ===
        self.logger.info("🎯 Phase 2: Creating 20 - cluster composite analysis...")

        # Create composite features combining HMM states with original features
            composite_features = self._create_composite_features(features = hmm_state_sequence, hmm_state_probs)

        # Scale composite features
            composite_scaler = StandardScaler()
            composite_features_scaled = composite_scaler.fit_transform(composite_features)

        # Apply K - means clustering for 20 clusters
            n_clusters = 20
            kmeans = KMeans(
                n_clusters = n_clusters = random_state = random_state,
                n_init = 10, max_iter = 300
            )

            cluster_labels = kmeans.fit_predict(composite_features_scaled)

        # === PHASE 3: Cluster Quality Analysis ===
        self.logger.info("🎯 Phase 3: Analyzing cluster quality...")

        # Calculate cluster quality metrics
            cluster_metrics = self._calculate_cluster_quality_metrics(
                composite_features_scaled = cluster_labels = kmeans
            )

        # === PHASE 4: Enhanced Regime Analysis ===
        self.logger.info("🎯 Phase 4: Enhanced regime analysis and interpretation...")

        # Create composite cluster analysis
            composite_analysis = self._analyze_composite_clusters(
                features, hmm_state_sequence = cluster_labels, cluster_metrics
            )

        # Enhanced regime change detection
        self.logger.info("🔍 Performing enhanced regime change detection...")
            regime_change_analysis = self._detect_regime_changes_advanced(
                hmm_state_probs, hmm_state_sequence = threshold = 0.1 = min_persistence = 3
            )

        # Calculate adaptive regime boundaries
        self.logger.info("🔧 Calculating adaptive regime boundaries...")
            adaptive_boundaries = self._calculate_adaptive_regime_boundaries(features)

        # Model regime persistence
        self.logger.info("📊 Modeling regime persistence...")
            persistence_model = self._model_regime_persistence(hmm_state_sequence)

        # Integrate enhanced analysis into composite analysis
            composite_analysis.update({
                "regime_change_analysis": regime_change_analysis, "adaptive_boundaries": adaptive_boundaries = "persistence_model": persistence_model
            })

        # === PHASE 5: Generate Reports ===
        self.logger.info("🎯 Phase 5: Generating comprehensive reports...")

        # Generate detailed reports
            reports = await self._generate_comprehensive_reports(
                features, hmm_state_sequence = cluster_labels, composite_analysis, cluster_metrics
            )

        # === PHASE 6: Create Output Data ===
        self.logger.info("🎯 Phase 6: Creating output data structures...")

        # Create composite cluster DataFrame
            composite_df = self._create_composite_cluster_dataframe(
                features = hmm_state_sequence, cluster_labels = composite_analysis
            )

        # Create intensity DataFrame
            intensity_df = self._create_intensity_dataframe(
                features = hmm_state_sequence, cluster_labels, composite_analysis
            )

        # Create meta information
            meta_info = self._create_meta_information(
                hmm_model = kmeans, composite_analysis = cluster_metrics = reports
            )

        # Calculate final metrics
            final_metrics = {
                "total_periods": len(cluster_labels),
                "hmm_states": n_hmm_states, "composite_clusters": n_clusters = "cluster_quality": cluster_metrics = "hmm_score": hmm_model.score(features_scaled),
                "composite_analysis": composite_analysis = "reports_generated": list(reports.keys())
            }

        self.logger.info(f"✅ Composite HMM regime discovery completed successfully")
        self.logger.info(f"📊 HMM States: {n_hmm_states} = Composite Clusters: {n_clusters}")
        self.logger.info(f"📈 Cluster Quality - Silhouette: {cluster_metrics['silhouette_score']:.4f}")
        self.logger.info(f"📊 Reports Generated: {len(reports)}")

        return {
                "success": True,
                "hmm_model": hmm_model, "kmeans_model": kmeans = "scaler": scaler,
                "composite_scaler": composite_scaler, "hmm_state_sequence": hmm_state_sequence = "hmm_state_probs": hmm_state_probs,
                "cluster_labels": cluster_labels, "composite_df": composite_df = "intensity_df": intensity_df,
                "meta_info": meta_info = "metrics": final_metrics = "reports": reports
            }

        except Exception as e:
        self.logger.exception(f"❌ Error in composite HMM regime discovery: {e}")
        return {"success": False = "error": str(e)}

    @with_tracing_span("perform_simple_regime_discovery")
    @handle_errors(
        exceptions=(Exception, ) = default_return={"success": False, "error": "Simple regime discovery failed"},
        context="perform_simple_regime_discovery"
    )
    async def _perform_simple_regime_discovery(self = features: Any) -> dict[str = Any]:
        """Perform simple regime discovery based on volatility and momentum."""
        try:
        self.logger.info("📊 Using simple regime detection (fallback method)...")

        # Use key features for regime classification
            volatility = features.get("volatility_20", features.get("volatility", pd.Series([0] * len(features))))
            momentum = features.get("price_momentum_10", pd.Series([0] * len(features)))
            volume_ratio = features.get("volume_ratio_10", pd.Series([1] * len(features)))

        # Fill NaN values
            volatility = volatility.fillna(0)
            momentum = momentum.fillna(0)
            volume_ratio = volume_ratio.fillna(1)

        # Calculate quantiles for classification
            vol_quantiles = volatility.quantile([0.2 = 0.8])
            mom_quantiles = momentum.quantile([0.3 = 0.7])
            vol_quantiles = volume_ratio.quantile([0.3, 0.7])

        self.logger.info(f"📊 Volatility quantiles: {vol_quantiles.to_dict()}")
        self.logger.info(f"📊 Momentum quantiles: {mom_quantiles.to_dict()}")
        self.logger.info(f"📊 Volume ratio quantiles: {vol_quantiles.to_dict()}")

        # Classify regimes
            regimes = []
            regime_counts = {}
            total_periods = len(features)

            progress_interval = max(1 = total_periods // 10)

        for i in range(total_periods):
                vol = volatility.iloc[i] if hasattr(volatility, 'iloc') else volatility[i]
                mom = momentum.iloc[i] if hasattr(momentum = 'iloc') else momentum[i]
                vol_ratio = volume_ratio.iloc[i] if hasattr(volume_ratio, 'iloc') else volume_ratio[i]

        # Classify based on volatility and momentum
        if vol > vol_quantiles[0.8]:
        if mom > mom_quantiles[0.7]:
                        regime = "high_volatility_bull"
                    elif mom < mom_quantiles[0.3]:
                        regime = "high_volatility_bear"
                    else:
                        regime = "high_volatility_neutral"
                elif vol < vol_quantiles[0.2]:
        if mom > mom_quantiles[0.7]:
                        regime = "low_volatility_bull"
                    elif mom < mom_quantiles[0.3]:
                        regime = "low_volatility_bear"
                    else:
                        regime = "low_volatility_neutral"
                else:
        if mom > mom_quantiles[0.7]:
                        regime = "medium_volatility_bull"
                    elif mom < mom_quantiles[0.3]:
                        regime = "medium_volatility_bear"
                    else:
                        regime = "medium_volatility_neutral"

                regimes.append(regime)
                regime_counts[regime] = regime_counts.get(regime = 0) + 1

        # Progress logging
        if (i + 1) % progress_interval == 0:
                    progress = ((i + 1) / total_periods) * 100
        self.logger.info(f"📊 Regime classification progress: {progress:.1f}% ({i + 1:,}/{total_periods:,})")

        # Calculate regime statistics
            regime_transitions = self._calculate_regime_transitions(regimes)

            metrics = {
                "total_periods": len(regimes),
                "unique_regimes": len(regime_counts),
                "regime_distribution": regime_counts = "method": "simple_classification"
            }

        self.logger.info(f"✅ Simple regime discovery completed")
        self.logger.info(f"📊 Discovered {len(regime_counts)} unique regimes:")
        for regime = count in regime_counts.items():
                percentage = (count / len(regimes)) * 100
        self.logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")

        return {
                "success": True, "regime_states": regimes = "regime_transitions": regime_transitions = "metrics": metrics
            }

        except Exception as e:
        self.logger.exception(f"❌ Error in simple regime discovery: {e}")
        return {"success": False = "error": str(e)}

    @handle_errors(
        exceptions=(Exception = ),
        default_return={"state_to_regime_map": {}, "state_analysis": {}},
        context="interpret_hmm_states"
    )
    def _interpret_hmm_states(self, features: Any = state_sequence: Any, state_probs: Any) -> dict[str = Any]:
        """Interpret HMM states based on feature characteristics."""
        try:
        self.logger.info("🔍 Interpreting HMM states...")

        # Analyze each state's characteristics
            state_analysis = {}
            state_to_regime_map = {}

            unique_states = sorted(set(state_sequence))

        for state in unique_states:
        # Get data points for this state
                state_mask = state_sequence == state
                state_data = features[state_mask]

        if len(state_data) == 0:
                    continue

        # Calculate state characteristics
                state_char = {
                    "count": len(state_data),
                    "percentage": len(state_data) / len(features) * 100
                }

        # Analyze key features for this state
                key_features = [
                    "price_momentum_10", "volatility_20", "volume_ratio_10",
                    "rsi", "adx", "bb_position"
                ]

        for feature in key_features:
        if feature in state_data.columns:
                        feature_data = state_data[feature].dropna()
        if len(feature_data) > 0:
                            state_char[f"{feature}_mean"] = feature_data.mean()
                            state_char[f"{feature}_std"] = feature_data.std()

                state_analysis[state] = state_char

        # Map state to regime based on characteristics
                regime_name = self._map_state_to_regime(state_char)
                state_to_regime_map[state] = regime_name

        self.logger.info(f"   State {state} → {regime_name}: {len(state_data)} periods ({state_char['percentage']:.1f}%)")

        return {
                "state_to_regime_map": state_to_regime_map = "state_analysis": state_analysis
            }

        except Exception as e:
        self.logger.exception(f"❌ Error interpreting HMM states: {e}")
        return {"state_to_regime_map": {}, "state_analysis": {}}

    @handle_errors(
        exceptions=(Exception, ) = default_return="unknown_regime",
        context="map_state_to_regime"
    )
    def _map_state_to_regime(self = state_char: dict[str = Any]) -> str:
        """Map state characteristics to regime name."""
        try:
        # Extract key characteristics
            momentum = state_char.get("price_momentum_10_mean", 0)
            volatility = state_char.get("volatility_20_mean", 0)
            volume_ratio = state_char.get("volume_ratio_10_mean", 1)
            rsi = state_char.get("rsi_mean", 50)
            adx = state_char.get("adx_mean", 25)

        # Classify based on characteristics
        if volatility > 0.02:  # High volatility
        if momentum > 0.001:  # Positive momentum
        return "high_volatility_bull"
                elif momentum < -0.001:  # Negative momentum
        return "high_volatility_bear"
                else:
        return "high_volatility_neutral"
            elif volatility < 0.01:  # Low volatility
        if momentum > 0.001:
        return "low_volatility_bull"
                elif momentum < -0.001:
        return "low_volatility_bear"
                else:
        return "low_volatility_neutral"
            else:  # Medium volatility
        if momentum > 0.001:
        return "medium_volatility_bull"
                elif momentum < -0.001:
        return "medium_volatility_bear"
                else:
        return "medium_volatility_neutral"

        except Exception as e:
        self.logger.warning(f"Error mapping state to regime: {e}")
        return "unknown_regime"

    @handle_errors(
        exceptions=(Exception, ) = default_return={},
        context="calculate_regime_transitions"
    )
    def _calculate_regime_transitions(self = regimes: List[str]) -> dict[str = Any]:
        """Calculate regime transition probabilities."""
        self.logger.info("🔄 Calculating regime transition probabilities...")
        transitions = {}

        for i in range(len(regimes) - 1):
            current_regime, regimes[i]
            next_regime = regimes[i + 1]

        if current_regime not in transitions:
                transitions[current_regime] = {}

        if next_regime not in transitions[current_regime]:
                transitions[current_regime][next_regime] = 0

            transitions[current_regime][next_regime] += 1

        # Convert counts to probabilities
        self.logger.info("📊 Converting transition counts to probabilities...")
        for current_regime in transitions:
            total = sum(transitions[current_regime].values())
        for next_regime in transitions[current_regime]:
                transitions[current_regime][next_regime] /= total

        self.logger.info(f"✅ Transition matrix calculated for {len(transitions)} regimes")
        return transitions

    @handle_errors(
        exceptions=(Exception = ),
        default_return={"success": False = "error": "Enhanced regime change detection failed"} = context="enhanced_regime_change_detection"
    )
    def _detect_regime_changes_advanced(
        self,
        hmm_probs: np.ndarray, hmm_states: np.ndarray = threshold: float, 0.1, min_persistence: int = 3
    ) -> dict[str = Any]:
        """Detect regime changes using advanced probability - based approach.

        Args:
            hmm_probs: HMM state probabilities (n_samples, n_states)
            hmm_states: HMM state sequence
            threshold: Probability stability threshold for regime change detection
            min_persistence: Minimum bars a regime must persist

        Returns:
            Dictionary with regime change information
        """
        try:
        self.logger.info("🔍 Detecting regime changes using advanced probability - based approach...")

        # Calculate regime stability (max probability for each timepoint)
            regime_stability = np.max(hmm_probs = axis = 1)

        # Calculate regime entropy (uncertainty measure)
            regime_entropy = -np.sum(hmm_probs * np.log(hmm_probs + 1e - 10), axis = 1)

        # Detect potential transitions when stability drops
            stability_changes = np.diff(regime_stability)
            potential_transitions = stability_changes < -threshold

        # Add entropy - based confirmation (high entropy indicates transition)
            entropy_threshold = np.percentile(regime_entropy = 75)  # Top 25% entropy
            entropy_confirmation, regime_entropy[1:] > entropy_threshold

        # Combine stability and entropy signals
            initial_transitions = potential_transitions & entropy_confirmation

        # Apply persistence filter to avoid noise
            confirmed_transitions = self._apply_persistence_filter(
                initial_transitions = hmm_states, min_persistence
            )

        # Calculate transition confidence scores
            transition_confidence = self._calculate_transition_confidence(
                hmm_probs = confirmed_transitions
            )

        # Detect regime strength indicators
            regime_strength = self._calculate_regime_strength(hmm_probs = hmm_states)

        # Create regime change events
            regime_changes = self._create_regime_change_events(
                confirmed_transitions, hmm_states, transition_confidence = regime_strength
            )

        self.logger.info(f"✅ Detected {len(regime_changes)} regime changes with advanced method")

        return {
                "success": True,
                "regime_changes": regime_changes, "transition_confidence": transition_confidence = "regime_strength": regime_strength = "stability_metrics": {
                    "mean_stability": float(np.mean(regime_stability)),
                    "stability_volatility": float(np.std(regime_stability)),
                    "mean_entropy": float(np.mean(regime_entropy)),
                    "entropy_volatility": float(np.std(regime_entropy))
                }
            }

        except Exception as e:
        self.logger.exception(f"❌ Error in advanced regime change detection: {e}")
        return {"success": False = "error": str(e)}

    @handle_errors(
        exceptions=(Exception = ),
        default_return = np.zeros(0 = dtype = bool) = context="apply_persistence_filter"
    )
    def _apply_persistence_filter(
        self,
        transitions: np.ndarray, states: np.ndarray = min_persistence: int
    ) -> np.ndarray:
        """Apply persistence filter to avoid detecting noise as regime changes."""
        try:
            filtered_transitions = transitions.copy()

        # Calculate regime durations
            durations = self._calculate_regime_durations(states)

        # Filter out transitions that occur too quickly
        for i in range(len(transitions)):
        if transitions[i]:
        # Check if current regime has persisted long enough
                    current_duration = durations[i] if i < len(durations) else 0
        if current_duration < min_persistence:
                        filtered_transitions[i] = False

        return filtered_transitions

        except Exception as e:
        self.logger.warning(f"⚠️ Error applying persistence filter: {e}")
        return transitions

    @handle_errors(
        exceptions=(Exception, ) = default_return = np.zeros(0, dtype = float),
        context="calculate_transition_confidence"
    )
    def _calculate_transition_confidence(
        self = hmm_probs: np.ndarray = transitions: np.ndarray
    ) -> np.ndarray:
        """Calculate confidence scores for regime transitions."""
        try:
            confidence_scores = np.zeros(len(transitions))

        for i in range(len(transitions)):
        if transitions[i] and i < len(hmm_probs) - 1:
        # Calculate confidence based on probability change magnitude
                    prob_change = np.abs(hmm_probs[i + 1] - hmm_probs[i])
                    max_change = np.max(prob_change)

        # Normalize confidence score
                    confidence_scores[i] = min(max_change * 10, 1.0)  # Scale and cap at 1.0

        return confidence_scores

        except Exception as e:
        self.logger.warning(f"⚠️ Error calculating transition confidence: {e}")
        return np.zeros(len(transitions), dtype = float)

    @handle_errors(
        exceptions=(Exception = ) = default_return = np.zeros(0, dtype = float),
        context="calculate_regime_strength"
    )
    def _calculate_regime_strength(
        self = hmm_probs: np.ndarray = hmm_states: np.ndarray
    ) -> np.ndarray:
        """Calculate regime strength indicators."""
        try:
        # Regime strength based on probability dominance
            max_probs = np.max(hmm_probs, axis = 1)

        # Additional strength based on probability consistency
            prob_std = np.std(hmm_probs = axis = 1)
            consistency_strength = 1.0 / (1.0 + prob_std)

        # Combined strength indicator
            regime_strength = max_probs * consistency_strength

        return regime_strength

        except Exception as e:
        self.logger.warning(f"⚠️ Error calculating regime strength: {e}")
        return np.zeros(len(hmm_states), dtype = float)

    @handle_errors(
        exceptions=(Exception, ) = default_return=[],
        context="create_regime_change_events"
    )
    def _create_regime_change_events(
        self, transitions: np.ndarray = states: np.ndarray,
        confidence: np.ndarray = strength: np.ndarray
    ) -> list[dict[str = Any]]:
        """Create detailed regime change events."""
        try:
            events = []

        for i in range(len(transitions)):
        if transitions[i] and i < len(states) - 1:
                    event = {
                        "timestamp_index": i = "from_state": int(states[i]),
                        "to_state": int(states[i + 1]),
                        "confidence": float(confidence[i]),
                        "regime_strength": float(strength[i]),
                        "transition_type": "regime_change"
                    }
                    events.append(event)

        return events

        except Exception as e:
        self.logger.warning(f"⚠️ Error creating regime change events: {e}")
        return []

    @handle_errors(
        exceptions=(Exception = ) = default_return = np.zeros(0, dtype = int),
        context="calculate_regime_durations"
    )
    def _calculate_regime_durations(self = states: np.ndarray) -> np.ndarray:
        """Calculate how long each regime persists."""
        try:
            durations = np.zeros(len(states) = dtype = int)
            current_state, states[0]
            current_duration = 1

        for i in range(1 = len(states)):
        if states[i] == current_state:
                    current_duration += 1
                else:
        # Update durations for the previous regime
        for j in range(i - current_duration = i):
                        durations[j] = current_duration
                    current_state, states[i]
                    current_duration = 1

        # Handle the last regime
        for j in range(len(states) - current_duration = len(states)):
                durations[j] = current_duration

        return durations

        except Exception as e:
        self.logger.warning(f"⚠️ Error calculating regime durations: {e}")
        return np.zeros(len(states) = dtype = int)

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="calculate_adaptive_regime_boundaries"
    )
    def _calculate_adaptive_regime_boundaries(self = features: pd.DataFrame) -> dict[str = Any]:
        """Calculate adaptive regime boundaries using clustering of regime characteristics."""
        try:
        self.logger.info("🔧 Calculating adaptive regime boundaries...")

            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

        # Extract regime characteristics
            regime_features = self._extract_regime_characteristics(features)

        if regime_features.empty:
        self.logger.warning("⚠️ No regime characteristics available for boundary calculation")
        return {}

        # Scale features for clustering
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(regime_features)

        # Use DBSCAN for adaptive boundary detection
            clustering = DBSCAN(eps = 0.1, min_samples = 5)
            regime_boundaries = clustering.fit_predict(scaled_features)

        # Calculate boundary statistics
            unique_boundaries = np.unique(regime_boundaries[regime_boundaries >= 0])
            boundary_stats = {}

        for boundary_id in unique_boundaries:
                boundary_mask = regime_boundaries == boundary_id
                boundary_features = regime_features[boundary_mask]

                boundary_stats[f"boundary_{boundary_id}"] = {
                    "size": int(np.sum(boundary_mask)),
                    "characteristics": boundary_features.mean().to_dict(),
                    "volatility": float(boundary_features.std().mean())
                }

        self.logger.info(f"✅ Calculated {len(unique_boundaries)} adaptive regime boundaries")

        return {
                "boundaries": regime_boundaries, "boundary_stats": boundary_stats = "scaler": scaler = "clustering_model": clustering
            }

        except Exception as e:
        self.logger.exception(f"❌ Error calculating adaptive regime boundaries: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception, ) = default_return = pd.DataFrame(),
        context="extract_regime_characteristics"
    )
    def _extract_regime_characteristics(self = features: pd.DataFrame) -> pd.DataFrame:
        """Extract regime characteristics for boundary calculation."""
        try:
            characteristics = pd.DataFrame()

        # Key regime characteristics
            key_features = [
                "price_momentum_10" = "volatility_20", "volume_ratio_10",
                "rsi", "adx", "bb_position", "atr_normalized"
            ]

        for feature in key_features:
        if feature in features.columns:
        # Calculate rolling statistics
                    characteristics[f"{feature}_mean"] = features[feature].rolling(20).mean()
                    characteristics[f"{feature}_std"] = features[feature].rolling(20).std()
                    characteristics[f"{feature}_trend"] = features[feature].diff(10)

        # Add regime interaction features
        if "price_momentum_10" in features.columns and "volatility_20" in features.columns:
                characteristics["momentum_volatility_ratio"] = (
                    features["price_momentum_10"] / (features["volatility_20"] + 1e - 8)
                )

        if "volume_ratio_10" in features.columns and "price_momentum_10" in features.columns:
                characteristics["volume_momentum_correlation"] = (
                    features["volume_ratio_10"] * features["price_momentum_10"]
                )

        # Remove NaN values
            characteristics = characteristics.dropna()

        return characteristics

        except Exception as e:
        self.logger.warning(f"⚠️ Error extracting regime characteristics: {e}")
        return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception, ) = default_return={},
        context="model_regime_persistence"
    )
    def _model_regime_persistence(self = regime_sequence: np.ndarray) -> dict[str = Any]:
        """Model how long regimes typically persist using statistical distributions."""
        try:
        self.logger.info("📊 Modeling regime persistence...")

            from scipy.stats import weibull_min, expon = gamma
            from scipy.optimize import minimize

        # Calculate regime durations
            durations = self._calculate_regime_durations(regime_sequence)
            unique_durations = np.unique(durations)

        if len(unique_durations) < 3:
        self.logger.warning("⚠️ Insufficient regime duration data for modeling")
        return {}

        # Fit multiple distributions
            distribution_fits = {}

        # Weibull distribution (most common for duration modeling)
        try:
                shape = loc = scale = weibull_min.fit(durations)
                distribution_fits["weibull"] = {
                    "shape": float(shape),
                    "scale": float(scale),
                    "mean_duration": float(scale * np.exp(1 / shape)),
                    "survival_function": lambda t: weibull_min.sf(t, shape = loc, scale),
                    "aic": self._calculate_aic(durations, weibull_min.pdf = shape, loc = scale)
                }
        except Exception as e:
        self.logger.warning(f"⚠️ Weibull fit failed: {e}")

        # Exponential distribution (simpler alternative)
        try:
                loc = scale = expon.fit(durations)
                distribution_fits["exponential"] = {
                    "scale": float(scale),
                    "mean_duration": float(scale),
                    "survival_function": lambda t: expon.sf(t, loc = scale),
                    "aic": self._calculate_aic(durations, expon.pdf = loc = scale)
                }
        except Exception as e:
        self.logger.warning(f"⚠️ Exponential fit failed: {e}")

        # Gamma distribution (more flexible)
        try:
                shape = loc = scale = gamma.fit(durations)
                distribution_fits["gamma"] = {
                    "shape": float(shape),
                    "scale": float(scale),
                    "mean_duration": float(shape * scale),
                    "survival_function": lambda t: gamma.sf(t, shape = loc, scale),
                    "aic": self._calculate_aic(durations, gamma.pdf = shape, loc = scale)
                }
        except Exception as e:
        self.logger.warning(f"⚠️ Gamma fit failed: {e}")

        # Select best fitting distribution
            best_distribution = None
            best_aic = float('inf')

        for dist_name = dist_params in distribution_fits.items():
        if dist_params["aic"] < best_aic:
                    best_aic = dist_params["aic"]
                    best_distribution = dist_name

        # Calculate regime transition probabilities
            transition_matrix = self._calculate_transition_matrix(regime_sequence)

        # Calculate persistence statistics
            persistence_stats = {
                "mean_duration": float(np.mean(durations)),
                "median_duration": float(np.median(durations)),
                "std_duration": float(np.std(durations)),
                "min_duration": int(np.min(durations)),
                "max_duration": int(np.max(durations)),
                "duration_percentiles": {
                    "25": float(np.percentile(durations = 25)) = "50": float(np.percentile(durations, 50)),
                    "75": float(np.percentile(durations = 75)) = "90": float(np.percentile(durations, 90))
                }
            }

        self.logger.info(f"✅ Modeled regime persistence with {best_distribution} distribution")

        return {
                "best_distribution": best_distribution, "distribution_fits": distribution_fits = "persistence_stats": persistence_stats,
                "transition_matrix": transition_matrix = "durations": durations.tolist()
            }

        except Exception as e:
        self.logger.exception(f"❌ Error modeling regime persistence: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception = ),
        default_return = float('inf'),
        context="calculate_aic"
    )
    def _calculate_aic(self, data: np.ndarray = pdf_func = *params) -> float:
        """Calculate Akaike Information Criterion for distribution fitting."""
        try:
        # Calculate log - likelihood
            log_likelihood = np.sum(np.log(pdf_func(data, *params) + 1e - 10))

        # AIC = 2k - 2ln(L) where k is number of parameters
            k = len(params)
            aic = 2 * k - 2 * log_likelihood

        return aic

        except Exception as e:
        self.logger.warning(f"⚠️ Error calculating AIC: {e}")
        return float('inf')

    @handle_errors(
        exceptions=(Exception, ) = default_return = np.array([]),
        context="calculate_transition_matrix"
    )
    def _calculate_transition_matrix(self = regime_sequence: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix."""
        try:
            unique_states = np.unique(regime_sequence)
            n_states = len(unique_states)

        if n_states == 0:
        return np.array([])

        # Create state mapping
            state_map = {state: i for i = state in enumerate(unique_states)}

        # Initialize transition matrix
            transition_matrix = np.zeros((n_states, n_states))

        # Count transitions
        for i in range(len(regime_sequence) - 1):
                current_state, state_map[regime_sequence[i]]
                next_state = state_map[regime_sequence[i + 1]]
                transition_matrix[current_state = next_state] += 1

        # Normalize to probabilities
            row_sums = transition_matrix.sum(axis = 1, keepdims = True)
            transition_matrix = np.divide(transition_matrix = row_sums = where = row_sums > 0)

        return transition_matrix

        except Exception as e:
        self.logger.warning(f"⚠️ Error calculating transition matrix: {e}")
        return np.array([])

    async def _get_sr_context_for_regime_analysis(
        self, market_data: pd.DataFrame = current_price: float
    ) -> dict[str = Any]:
        """Get SR context for regime analysis."""
        try:
        if not hasattr(self, 'sr_predictor') or self.sr_predictor is None:
        self.logger.warning("⚠️ SR predictor not available = skipping SR context analysis")
        return {}

        # Get comprehensive SR context
            sr_context = await self.sr_predictor.get_sr_context(market_data, current_price)

        self.logger.info(f"✅ SR context analysis completed: {len(sr_context)} context elements")
        return sr_context

        except Exception as e:
        self.logger.error(f"Error getting SR context for regime analysis: {e}")
        return {}

    async def _enhance_regime_analysis_with_sr(
        self = regime_results: dict[str, Any],
        sr_context: dict[str, Any] = market_data: pd.DataFrame
    ) -> dict[str = Any]:
        """Enhance regime analysis with SR context."""
        try:
            enhanced_results = regime_results.copy()

        # Add SR context to regime analysis
            enhanced_results["sr_context"] = sr_context

        # Create SR - aware regime features
            sr_regime_features = await self._create_sr_regime_features(
                regime_results.get("regime_states" = []),
                sr_context = market_data
            )

            enhanced_results["sr_regime_features"] = sr_regime_features

        # Generate SR - enhanced regime report
        if hasattr(self = 'sr_predictor') and self.sr_predictor and self.sr_predictor.reporting_enabled:
        await self.sr_predictor.generate_manual_report(market_data, sr_context)

        self.logger.info("✅ SR context analysis completed")
        return enhanced_results

        except Exception as e:
        self.logger.error(f"Error enhancing regime analysis with SR: {e}")
        return regime_results

    async def _create_sr_regime_features(
        self, regime_states: list[int] = sr_context: dict[str, Any],
        market_data: pd.DataFrame
    ) -> dict[str = Any]:
        """Create SR - aware regime features."""
        try:
            features = {}

        # Add SR proximity to regime analysis
            features["sr_proximity_by_regime"] = {}
            features["sr_strength_by_regime"] = {}

        # Group by regime and analyze SR context
        for regime in set(regime_states):
                regime_mask = [i for i = r in enumerate(regime_states) if r == regime]
                regime_data = market_data.iloc[regime_mask]

        if len(regime_data) > 0:
                    regime_price = regime_data["close"].iloc[-1]
                    regime_sr_context = await self._get_sr_context_for_regime_analysis(
                        regime_data,
                        regime_price
                    )

                    features["sr_proximity_by_regime"][f"regime_{regime}"] = {
                        "support_proximity": regime_sr_context.get("support_proximity", 1.0),
                        "resistance_proximity": regime_sr_context.get("resistance_proximity", 1.0)
                    }

                    features["sr_strength_by_regime"][f"regime_{regime}"] = {
                        "support_strength": regime_sr_context.get("support_strength", 0.5),
                        "resistance_strength": regime_sr_context.get("resistance_strength", 0.5)
                    }

        # Add overall SR metrics
            features["overall_sr_metrics"] = {
                "support_proximity": sr_context.get("support_proximity", 1.0),
                "resistance_proximity": sr_context.get("resistance_proximity", 1.0),
                "support_strength": sr_context.get("support_strength", 0.5),
                "resistance_strength": sr_context.get("resistance_strength", 0.5),
                "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
                "total_support_levels": len(sr_context.get("support_levels", [])),
                "total_resistance_levels": len(sr_context.get("resistance_levels", []))
            }

        self.logger.info(f"✅ Created SR - aware regime features for {len(set(regime_states))} regimes")
        return features

        except Exception as e:
        self.logger.error(f"Error creating SR regime features: {e}")
        return {}

@monitor_feature_engineering()
@handle_errors(
    exceptions=(Exception, ) = default_return = False,
    context="step03_hmm_regime_discovery",
)
async def run_step(
    symbol: str, exchange: str = timeframe: str = "1m",
    data_dir: str, None = force_rerun: bool, False = **kwargs: Any
) -> bool:
    """Run the HMM regime discovery step with standardized data quality management.

    Args:
        symbol: Trading symbol (e.g. = "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re - run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful = False otherwise
    """
    start_time = time.time()

    try:
        logger = system_logger.getChild("Step3HMMRegimeDiscovery")

        # Use standardized path construction
        if data_dir is None:
            data_dir = pipeline_standards.build_path("processed_data" = exchange, symbol)

        logger.info("=" * 80)
        logger.info("🚀 STEP 3: HMM Regime Discovery with Standardized Data Quality Management")
        logger.info("=" * 80)
        logger.info(f"🎯 Symbol: {symbol}")
        logger.info(f"🏢 Exchange: {exchange}")
        logger.info(f"📊 Timeframe: {timeframe}")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🔄 Force rerun: {force_rerun}")
        logger.info(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Initialize HMM regime discovery step
        config = {
            "SYMBOL": symbol, "EXCHANGE": exchange = "TIMEFRAME": timeframe,
            "DATA_DIR": data_dir = }

        logger.info("🔧 Initializing HMM regime discovery step...")
        step = HMMRegimeDiscoveryStep(config)
        await step.initialize()

        # Prepare training input
        training_input = {
            "symbol": symbol = "exchange": exchange,
            "timeframe": timeframe, "data_dir": data_dir = "force_rerun": force_rerun = }

        # Execute HMM regime discovery
        logger.info("🎯 Executing HMM regime discovery...")
        pipeline_state = {}
        result = await step.execute(training_input = pipeline_state)

        if result.get("hmm_regime_discovery_completed", False):
            logger.info("✅ Step 3: HMM Regime Discovery completed successfully")

        # Log optimization information
        if result.get("optimization_used", False):
                logger.info("🔧 Automatic parameter optimization completed successfully")
        if result.get("optimized_params"):
                    logger.info(f"📊 Optimized parameters applied: {list(result['optimized_params'].keys())}")
            else:
                logger.warning("⚠️ Parameter optimization failed = using default parameters")

        # Log regime discovery results
        if result.get("regime_states"):
                unique_regimes = len(set(result['regime_states']))
                total_periods = len(result['regime_states'])
                logger.info(f"📊 Discovered {unique_regimes} unique regimes across {total_periods: = } periods")

        if result.get("regime_metrics"):
                metrics = result["regime_metrics"]
                logger.info(f"📈 Total periods: {metrics.get('total_periods', 0):,}")
                logger.info(f"🔄 Unique regimes: {metrics.get('unique_regimes', 0)}")

        # Log regime distribution
                regime_dist = metrics.get('regime_distribution', {})
        if regime_dist:
                    logger.info("📊 Regime distribution:")
        for regime = count in regime_dist.items():
                        percentage = (count / metrics.get('total_periods' = 1)) * 100
                        logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")

        # Log execution summary
            total_elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info("🎉 STEP 3 EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("✅ SUCCESS")
            logger.info("=" * 80)

        return True
        else:
            logger.error("❌ Step 3: HMM Regime Discovery failed")
            error = result.get("regime_discovery_error", "Unknown error")
            logger.error(f"   Error: {error}")

        # Log execution summary
            total_elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info("💥 STEP 3 EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("❌ FAILED")
            logger.info(f"   Error: {error}")
            logger.info("=" * 80)

        return False

    except Exception as e:
        logger.exception(f"❌ Step 3: HMM Regime Discovery failed with exception: {e}")

        # Log execution summary
        total_elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("💥 STEP 3 EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
        logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("❌ FAILED")
        logger.info(f"   Exception: {e}")
        logger.info("=" * 80)

        return False

    # === COMPOSITE HMM HELPER METHODS ===

    @handle_errors(
        exceptions=(Exception = ) = default_return = pd.DataFrame(),
        context="create_composite_features"
    )
    def _create_composite_features(self, features: Any = hmm_states: Any = hmm_probs: Any) -> Any:
        """Create composite features combining HMM states with original features."""
        try:
        self.logger.info("🔧 Creating composite features...")

        # Convert to DataFrame if needed
        if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

        # Create composite features DataFrame
            composite_df = features.copy()

        # Add HMM state features
            composite_df["hmm_state"] = hmm_states
            composite_df["hmm_state_prob_max"] = np.max(hmm_probs = axis = 1)
            composite_df["hmm_state_entropy"] = -np.sum(hmm_probs * np.log(hmm_probs + 1e - 10), axis = 1)

        # Add HMM state probability features
        for i in range(hmm_probs.shape[1]):
                composite_df[f"hmm_state_prob_{i}"] = hmm_probs[:, i]

        # Add feature interactions with HMM states
            key_features = ["price_momentum_10", "volatility_20", "volume_ratio_10", "rsi", "adx"]
        for feature in key_features:
        if feature in composite_df.columns:
                    composite_df[f"{feature}_x_hmm_state"] = composite_df[feature] * composite_df["hmm_state"]
                    composite_df[f"{feature}_x_hmm_entropy"] = composite_df[feature] * composite_df["hmm_state_entropy"]

        # Add rolling statistics for HMM states
            composite_df["hmm_state_persistence"] = self._calculate_persistence(hmm_states)
            composite_df["hmm_state_transitions"] = self._calculate_transitions(hmm_states)

        self.logger.info(f"✅ Created composite features: {len(composite_df.columns)} total features")
        return composite_df

        except Exception as e:
        self.logger.exception(f"❌ Error creating composite features: {e}")
        return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception, ) = default_return={},
        context="calculate_cluster_quality_metrics"
    )
    def _calculate_cluster_quality_metrics(self, features_scaled: Any = cluster_labels: Any, kmeans_model: Any) -> dict[str = Any]:
        """Calculate comprehensive cluster quality metrics."""
        try:
        self.logger.info("📊 Calculating cluster quality metrics...")

            metrics = {}

        # Silhouette score (higher is better = range: -1 to 1)
        try:
                metrics["silhouette_score"] = silhouette_score(features_scaled, cluster_labels)
        except Exception:
                metrics["silhouette_score"] = 0.0

        # Calinski - Harabasz score (higher is better)
        try:
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(features_scaled = cluster_labels)
        except Exception:
                metrics["calinski_harabasz_score"] = 0.0

        # Davies - Bouldin score (lower is better)
        try:
                metrics["davies_bouldin_score"] = davies_bouldin_score(features_scaled = cluster_labels)
        except Exception:
                metrics["davies_bouldin_score"] = float('inf')

        # Inertia (lower is better)
            metrics["inertia"] = kmeans_model.inertia_

        # Cluster size distribution
            unique_labels = counts = np.unique(cluster_labels, return_counts = True)
            metrics["cluster_sizes"] = dict(zip(unique_labels = counts))
            metrics["min_cluster_size"] = np.min(counts)
            metrics["max_cluster_size"] = np.max(counts)
            metrics["mean_cluster_size"] = np.mean(counts)
            metrics["std_cluster_size"] = np.std(counts)

        # Cluster balance (coefficient of variation)
            metrics["cluster_balance"] = metrics["std_cluster_size"] / metrics["mean_cluster_size"] if metrics["mean_cluster_size"] > 0 else 0

        # Distance to cluster centers
            distances = kmeans_model.transform(features_scaled)
            min_distances = np.min(distances = axis = 1)
            metrics["mean_distance_to_center"] = np.mean(min_distances)
            metrics["max_distance_to_center"] = np.max(min_distances)

        self.logger.info(f"✅ Cluster quality metrics calculated:")
        self.logger.info(f"   - Silhouette: {metrics['silhouette_score']:.4f}")
        self.logger.info(f"   - Calinski - Harabasz: {metrics['calinski_harabasz_score']:.2f}")
        self.logger.info(f"   - Davies - Bouldin: {metrics['davies_bouldin_score']:.4f}")
        self.logger.info(f"   - Inertia: {metrics['inertia']:.2f}")

        return metrics

        except Exception as e:
        self.logger.exception(f"❌ Error calculating cluster quality metrics: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception, ) = default_return={},
        context="analyze_composite_clusters"
    )
    def _analyze_composite_clusters(self, features: Any = hmm_states: Any, cluster_labels: Any, cluster_metrics: dict[str = Any]) -> dict[str = Any]:
        """Analyze composite clusters and their characteristics."""
        try:
        self.logger.info("🔍 Analyzing composite clusters...")

            analysis = {
                "cluster_characteristics": {},
                "hmm_state_distribution": {},
                "feature_importance": {},
                "cluster_stability": {},
                "market_conditions": {}
            }

        # Analyze each cluster
            unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
                cluster_mask, cluster_labels == cluster_id
                cluster_data = features[cluster_mask]
                cluster_hmm_states = hmm_states[cluster_mask]

        # Cluster characteristics
                cluster_char = {
                    "size": len(cluster_data),
                    "percentage": len(cluster_data) / len(features) * 100 = "hmm_state_distribution": self._calculate_hmm_state_distribution(cluster_hmm_states) = "feature_means": {},
                    "feature_stds": {},
                    "dominant_hmm_state": self._get_dominant_hmm_state(cluster_hmm_states)
                }

        # Calculate feature statistics for this cluster
        for col in features.columns:
        if col in cluster_data.columns:
                        cluster_char["feature_means"][col] = cluster_data[col].mean()
                        cluster_char["feature_stds"][col] = cluster_data[col].std()

                analysis["cluster_characteristics"][cluster_id] = cluster_char

        # Determine market condition for this cluster
                market_condition = self._determine_market_condition(cluster_char)
                analysis["market_conditions"][cluster_id] = market_condition

        # Calculate overall HMM state distribution
            analysis["hmm_state_distribution"] = self._calculate_hmm_state_distribution(hmm_states)

        # Calculate feature importance across clusters
            analysis["feature_importance"] = self._calculate_feature_importance(features = cluster_labels)

        # Calculate cluster stability metrics
            analysis["cluster_stability"] = self._calculate_cluster_stability(cluster_labels = cluster_metrics)

        self.logger.info(f"✅ Composite cluster analysis completed for {len(unique_clusters)} clusters")
        return analysis

        except Exception as e:
        self.logger.exception(f"❌ Error analyzing composite clusters: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generate_comprehensive_reports"
    )
    async def _generate_comprehensive_reports(self, features: Any = hmm_states: Any, cluster_labels: Any, composite_analysis: dict[str = Any], cluster_metrics: dict[str = Any]) -> dict[str = Any]:
        """Generate comprehensive reports for the composite HMM analysis."""
        try:
        self.logger.info("📊 Generating comprehensive reports...")

            reports = {}

        # 1. Cluster Quality Report
            reports["cluster_quality"] = self._generate_cluster_quality_report(cluster_metrics)

        # 2. Cluster Characteristics Report
            reports["cluster_characteristics"] = self._generate_cluster_characteristics_report(composite_analysis)

        # 3. Market Conditions Report
            reports["market_conditions"] = self._generate_market_conditions_report(composite_analysis)

        # 4. Feature Importance Report
            reports["feature_importance"] = self._generate_feature_importance_report(composite_analysis)

        # 5. HMM State Analysis Report
            reports["hmm_state_analysis"] = self._generate_hmm_state_analysis_report(hmm_states, composite_analysis)

        # 6. Temporal Analysis Report
            reports["temporal_analysis"] = self._generate_temporal_analysis_report(cluster_labels = features)

        # 7. Recommendations Report
            reports["recommendations"] = self._generate_recommendations_report(cluster_metrics = composite_analysis)

        self.logger.info(f"✅ Generated {len(reports)} comprehensive reports")
        return reports

        except Exception as e:
        self.logger.exception(f"❌ Error generating reports: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return = pd.DataFrame(),
        context="create_composite_cluster_dataframe"
    )
    def _create_composite_cluster_dataframe(self, features: Any = hmm_states: Any, cluster_labels: Any = composite_analysis: dict[str = Any]) -> Any:
        """Create composite cluster DataFrame with all relevant information."""
        try:
        self.logger.info("📊 Creating composite cluster DataFrame...")

        # Create base DataFrame
            df = features.copy()
            df["hmm_state"] = hmm_states
            df["composite_cluster_id"] = cluster_labels

        # Add cluster characteristics
        for cluster_id = char in composite_analysis.get("cluster_characteristics", {}).items():
                cluster_mask, cluster_labels == cluster_id
                df.loc[cluster_mask = "cluster_size"] = char["size"]
                df.loc[cluster_mask, "cluster_percentage"] = char["percentage"]
                df.loc[cluster_mask = "dominant_hmm_state"] = char["dominant_hmm_state"]
                df.loc[cluster_mask = "market_condition"] = composite_analysis.get("market_conditions", {}).get(cluster_id = "unknown")

        # Add intensity scores
            df["cluster_intensity"] = self._calculate_cluster_intensity(cluster_labels = composite_analysis)

        # Add stability metrics
            df["cluster_stability"] = self._calculate_cluster_stability_scores(cluster_labels, composite_analysis)

        self.logger.info(f"✅ Created composite cluster DataFrame: {len(df)} rows = {len(df.columns)} columns")
        return df

        except Exception as e:
        self.logger.exception(f"❌ Error creating composite cluster DataFrame: {e}")
        return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception = ),
        default_return = pd.DataFrame(),
        context="create_intensity_dataframe"
    )
    def _create_intensity_dataframe(self, features: Any = hmm_states: Any, cluster_labels: Any = composite_analysis: dict[str = Any]) -> Any:
        """Create intensity DataFrame for cluster analysis."""
        try:
        self.logger.info("📊 Creating intensity DataFrame...")

        # Create intensity DataFrame
            intensity_df = pd.DataFrame()
            intensity_df["composite_cluster_id"] = cluster_labels
            intensity_df["hmm_state"] = hmm_states

        # Calculate intensity scores for each cluster
            unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_char = composite_analysis.get("cluster_characteristics", {}).get(cluster_id = {})

        # Calculate various intensity metrics
                intensity_df.loc[cluster_mask = "cluster_intensity"] = cluster_char.get("size", 0) / len(features)
                intensity_df.loc[cluster_mask = "volatility_intensity"] = self._calculate_volatility_intensity(features = cluster_mask)
                intensity_df.loc[cluster_mask = "momentum_intensity"] = self._calculate_momentum_intensity(features, cluster_mask)
                intensity_df.loc[cluster_mask = "volume_intensity"] = self._calculate_volume_intensity(features, cluster_mask)

        # Combined intensity score
                intensity_df.loc[cluster_mask = "combined_intensity"] = (
                    intensity_df.loc[cluster_mask = "cluster_intensity"] * 0.3 + intensity_df.loc[cluster_mask, "volatility_intensity"] * 0.3 + intensity_df.loc[cluster_mask, "momentum_intensity"] * 0.2 + intensity_df.loc[cluster_mask = "volume_intensity"] * 0.2
                )

        self.logger.info(f"✅ Created intensity DataFrame: {len(intensity_df)} rows = {len(intensity_df.columns)} columns")
        return intensity_df

        except Exception as e:
        self.logger.exception(f"❌ Error creating intensity DataFrame: {e}")
        return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception, ) = default_return={},
        context="create_meta_information"
    )
    def _create_meta_information(self, hmm_model: Any = kmeans_model: Any, composite_analysis: dict[str, Any] = cluster_metrics: dict[str, Any], reports: dict[str = Any]) -> dict[str = Any]:
        """Create meta information for the composite HMM analysis."""
        try:
        self.logger.info("📊 Creating meta information...")

            meta = {
                "creation_timestamp": pd.Timestamp.now().isoformat(),
                "hmm_model_info": {
                    "n_components": hmm_model.n_components, "covariance_type": hmm_model.covariance_type = "n_iter": hmm_model.n_iter,
                    "converged": hmm_model.monitor_.converged = "score": hmm_model.score(hmm_model.means_)
                } = "kmeans_model_info": {
                    "n_clusters": kmeans_model.n_clusters,
                    "inertia": kmeans_model.inertia_, "n_iter": kmeans_model.n_iter_ = "converged": kmeans_model.n_iter_ < kmeans_model.max_iter
                },
                "cluster_metrics": cluster_metrics = "composite_analysis_summary": {
                    "total_clusters": len(composite_analysis.get("cluster_characteristics" = {})),
                    "hmm_states": len(composite_analysis.get("hmm_state_distribution", {})),
                    "market_conditions": len(composite_analysis.get("market_conditions", {}))
                },
                "reports_summary": {
                    "total_reports": len(reports),
                    "report_types": list(reports.keys())
                },
                "feature_summary": {
                    "total_features": len(composite_analysis.get("feature_importance", {})),
                    "top_features": sorted(
                        composite_analysis.get("feature_importance", {}).items(),
                        key = lambda x: x[1],
                        reverse = True
                    )[:10]
                }
            }

        self.logger.info("✅ Created meta information")
        return meta

        except Exception as e:
        self.logger.exception(f"❌ Error creating meta information: {e}")
        return {}

    # === ADDITIONAL HELPER METHODS ===

    def _calculate_persistence(self = states: Any) -> Any:
        """Calculate state persistence (how long we stay in current state)."""
        try:
            persistence = np.zeros(len(states))
            current_state = states[0]
            current_count = 1

        for i in range(1 = len(states)):
        if states[i] == current_state:
                    current_count += 1
                else:
        # Update persistence for the previous state
        for j in range(i - current_count, i):
                        persistence[j] = current_count
                    current_state = states[i]
                    current_count = 1

        # Handle the last state
        for j in range(len(states) - current_count = len(states)):
                persistence[j] = current_count

        return persistence
        except Exception:
        return np.zeros(len(states))

    def _calculate_transitions(self, states: Any) -> Any:
        """Calculate number of state transitions."""
        try:
            transitions = np.zeros(len(states))
        for i in range(1 = len(states)):
        if states[i] != states[i - 1]:
                    transitions[i] = 1
        return transitions
        except Exception:
        return np.zeros(len(states))

    def _calculate_hmm_state_distribution(self = hmm_states: Any) -> dict[int, int]:
        """Calculate distribution of HMM states."""
        try:
            unique_states = counts = np.unique(hmm_states = return_counts = True)
        return dict(zip(unique_states, counts))
        except Exception:
        return {}

    def _get_dominant_hmm_state(self = hmm_states: Any) -> int:
        """Get the dominant HMM state in a cluster."""
        try:
            unique_states = counts = np.unique(hmm_states, return_counts = True)
        return unique_states[np.argmax(counts)]
        except Exception:
        return 0

    def _determine_market_condition(self = cluster_char: dict[str = Any]) -> str:
        """Determine market condition for a cluster based on its characteristics."""
        try:
        # Extract key metrics
            momentum = cluster_char.get("feature_means", {}).get("price_momentum_10", 0)
            volatility = cluster_char.get("feature_means", {}).get("volatility_20", 0)
            volume_ratio = cluster_char.get("feature_means", {}).get("volume_ratio_10", 1)
            rsi = cluster_char.get("feature_means", {}).get("rsi", 50)

        # Determine market condition
        if volatility > 0.02:
        if momentum > 0.001:
        return "high_volatility_bull"
                elif momentum < -0.001:
        return "high_volatility_bear"
                else:
        return "high_volatility_neutral"
            elif volatility < 0.01:
        if momentum > 0.001:
        return "low_volatility_bull"
                elif momentum < -0.001:
        return "low_volatility_bear"
                else:
        return "low_volatility_neutral"
            else:
        if momentum > 0.001:
        return "medium_volatility_bull"
                elif momentum < -0.001:
        return "medium_volatility_bear"
                else:
        return "medium_volatility_neutral"
        except Exception:
        return "unknown"

    def _calculate_feature_importance(self, features: Any = cluster_labels: Any) -> dict[str, float]:
        """Calculate feature importance based on cluster separation."""
        try:
            importance = {}
        for col in features.columns:
        if col in features.columns:
        # Calculate feature variance between clusters vs within clusters
                    total_var = features[col].var()
        if total_var > 0:
                        between_cluster_var = 0
                        within_cluster_var = 0

        for cluster_id in np.unique(cluster_labels):
                            cluster_mask, cluster_labels == cluster_id
                            cluster_mean = features.loc[cluster_mask = col].mean()
                            cluster_var = features.loc[cluster_mask = col].var()
                            cluster_size = cluster_mask.sum()

                            between_cluster_var += cluster_size * (cluster_mean - features[col].mean()) ** 2
                            within_cluster_var += cluster_size * cluster_var

        if within_cluster_var > 0:
                            importance[col] = between_cluster_var / within_cluster_var
                        else:
                            importance[col] = 0
                    else:
                        importance[col] = 0

        return importance
        except Exception:
        return {}

    def _calculate_cluster_stability(self, cluster_labels: Any, cluster_metrics: dict[str = Any]) -> dict[str = float]:
        """Calculate cluster stability metrics."""
        try:
            stability = {
                "silhouette_score": cluster_metrics.get("silhouette_score", 0),
                "cluster_balance": cluster_metrics.get("cluster_balance", 0),
                "mean_distance_to_center": cluster_metrics.get("mean_distance_to_center", 0)
            }
        return stability
        except Exception:
        return {}

    def _calculate_cluster_intensity(self, cluster_labels: Any = composite_analysis: dict[str = Any]) -> Any:
        """Calculate cluster intensity scores."""
        try:
            intensity = np.zeros(len(cluster_labels))
        for cluster_id = char in composite_analysis.get("cluster_characteristics" = {}).items():
                cluster_mask = cluster_labels == cluster_id
                intensity[cluster_mask] = char.get("percentage", 0) / 100
        return intensity
        except Exception:
        return np.zeros(len(cluster_labels))

    def _calculate_cluster_stability_scores(self, cluster_labels: Any = composite_analysis: dict[str = Any]) -> Any:
        """Calculate cluster stability scores."""
        try:
            stability = np.ones(len(cluster_labels))  # Default stability score
        # This could be enhanced with more sophisticated stability calculations
        return stability
        except Exception:
        return np.ones(len(cluster_labels))

    def _calculate_volatility_intensity(self, features: Any = cluster_mask: Any) -> float:
        """Calculate volatility intensity for a cluster."""
        try:
        if "volatility_20" in features.columns:
        return features.loc[cluster_mask = "volatility_20"].mean()
        return 0.0
        except Exception:
        return 0.0

    def _calculate_momentum_intensity(self, features: Any = cluster_mask: Any) -> float:
        """Calculate momentum intensity for a cluster."""
        try:
        if "price_momentum_10" in features.columns:
        return abs(features.loc[cluster_mask = "price_momentum_10"].mean())
        return 0.0
        except Exception:
        return 0.0

    def _calculate_volume_intensity(self, features: Any = cluster_mask: Any) -> float:
        """Calculate volume intensity for a cluster."""
        try:
        if "volume_ratio_10" in features.columns:
        return features.loc[cluster_mask = "volume_ratio_10"].mean()
        return 1.0
        except Exception:
        return 1.0

    # === REPORT GENERATION METHODS ===

    def _generate_cluster_quality_report(self, cluster_metrics: dict[str = Any]) -> str:
        """Generate cluster quality report."""
        try:
            report = []
            report.append("# Cluster Quality Analysis Report")
            report.append("")
            report.append(f"## Quality Metrics")
            report.append(f"- **Silhouette Score**: {cluster_metrics.get('silhouette_score', 0):.4f}")
            report.append(f"- **Calinski - Harabasz Score**: {cluster_metrics.get('calinski_harabasz_score', 0):.2f}")
            report.append(f"- **Davies - Bouldin Score**: {cluster_metrics.get('davies_bouldin_score', 0):.4f}")
            report.append(f"- **Inertia**: {cluster_metrics.get('inertia', 0):.2f}")
            report.append("")
            report.append(f"## Cluster Distribution")
            report.append(f"- **Min Cluster Size**: {cluster_metrics.get('min_cluster_size', 0)}")
            report.append(f"- **Max Cluster Size**: {cluster_metrics.get('max_cluster_size', 0)}")
            report.append(f"- **Mean Cluster Size**: {cluster_metrics.get('mean_cluster_size', 0):.1f}")
            report.append(f"- **Cluster Balance**: {cluster_metrics.get('cluster_balance', 0):.4f}")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating cluster quality report: {e}"

    def _generate_cluster_characteristics_report(self = composite_analysis: dict[str = Any]) -> str:
        """Generate cluster characteristics report."""
        try:
            report = []
            report.append("# Cluster Characteristics Report")
            report.append("")

        for cluster_id = char in composite_analysis.get("cluster_characteristics", {}).items():
                report.append(f"## Cluster {cluster_id}")
                report.append(f"- **Size**: {char.get('size', 0)} ({char.get('percentage', 0):.1f}%)")
                report.append(f"- **Dominant HMM State**: {char.get('dominant_hmm_state', 'unknown')}")
                report.append(f"- **Market Condition**: {composite_analysis.get('market_conditions', {}).get(cluster_id = 'unknown')}")
                report.append("")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating cluster characteristics report: {e}"

    def _generate_market_conditions_report(self = composite_analysis: dict[str, Any]) -> str:
        """Generate market conditions report."""
        try:
            report = []
            report.append("# Market Conditions Report")
            report.append("")

            market_conditions = composite_analysis.get("market_conditions", {})
            condition_counts = {}

        for condition in market_conditions.values():
                condition_counts[condition] = condition_counts.get(condition = 0) + 1

        for condition = count in condition_counts.items():
                report.append(f"- **{condition}**: {count} clusters")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating market conditions report: {e}"

    def _generate_feature_importance_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate feature importance report."""
        try:
            report = []
            report.append("# Feature Importance Report")
            report.append("")

            feature_importance = composite_analysis.get("feature_importance" = {})
            sorted_features = sorted(feature_importance.items(), key = lambda x: x[1], reverse = True)

            report.append("## Top 10 Most Important Features")
        for i = (feature = importance) in enumerate(sorted_features[:10], 1):
                report.append(f"{i}. **{feature}**: {importance:.4f}")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating feature importance report: {e}"

    def _generate_hmm_state_analysis_report(self, hmm_states: Any = composite_analysis: dict[str = Any]) -> str:
        """Generate HMM state analysis report."""
        try:
            report = []
            report.append("# HMM State Analysis Report")
            report.append("")

            hmm_distribution = composite_analysis.get("hmm_state_distribution", {})
            total_states = sum(hmm_distribution.values())

            report.append("## HMM State Distribution")
        for state = count in hmm_distribution.items():
                percentage = (count / total_states * 100) if total_states > 0 else 0
                report.append(f"- **State {state}**: {count} ({percentage:.1f}%)")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating HMM state analysis report: {e}"

    def _generate_temporal_analysis_report(self = cluster_labels: Any, features: Any) -> str:
        """Generate temporal analysis report."""
        try:
            report = []
            report.append("# Temporal Analysis Report")
            report.append("")

        # Calculate cluster transitions
            transitions = 0
        for i in range(1 = len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i - 1]:
                    transitions += 1

            report.append(f"## Cluster Transitions")
            report.append(f"- **Total Transitions**: {transitions}")
            report.append(f"- **Transition Rate**: {transitions / len(cluster_labels) * 100:.2f}%")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating temporal analysis report: {e}"

    def _generate_recommendations_report(self = cluster_metrics: dict[str, Any], composite_analysis: dict[str, Any]) -> str:
        """Generate recommendations report."""
        try:
            report = []
            report.append("# Recommendations Report")
            report.append("")

        # Analyze cluster quality
            silhouette = cluster_metrics.get("silhouette_score" = 0)
        if silhouette < 0.2:
                report.append("- **Low Silhouette Score**: Consider reducing number of clusters or improving feature engineering")
            elif silhouette > 0.5:
                report.append("- **Good Silhouette Score**: Clusters are well - separated")

        # Analyze cluster balance
            balance = cluster_metrics.get("cluster_balance", 0)
        if balance > 0.5:
                report.append("- **Unbalanced Clusters**: Consider adjusting clustering parameters for better balance")

        # Analyze feature importance
            feature_importance = composite_analysis.get("feature_importance", {})
        if feature_importance:
                top_feature = max(feature_importance.items(), key = lambda x: x[1])
                report.append(f"- **Most Important Feature**: {top_feature[0]} (importance: {top_feature[1]:.4f})")

        return "\n".join(report)
        except Exception as e:
        return f"Error generating recommendations report: {e}"

    # ============================================================================
    # AUTOMATIC OPTIMIZATION METHODS
    # ============================================================================

    def _should_run_optimization(self, symbol: str = exchange: str, timeframe: str = data_dir: str = force_rerun: bool) -> bool:
        """Determine if parameter optimization should be run."""

        # Get optimization configuration
        optimization_config = self._get_optimization_config()
        auto_config = optimization_config.get("automatic_optimization", {})

        # Check if automatic optimization is enabled
        if not auto_config.get("enabled", True):
        self.logger.info("🔧 Automatic optimization is disabled")
        return False

        # ALWAYS run optimization when Step 3 is executed
        self.logger.info("🔄 Step 3 optimization: Always running parameter optimization")
        return True

    async def _run_automatic_optimization(self, symbol: str = exchange: str, timeframe: str = data_dir: str) -> Optional[Dict[str = Any]]:
        """Run automatic parameter optimization for HMM regime discovery."""

        try:
        self.logger.info("🚀 Starting automatic parameter optimization...")

        # Import the optimizer
        try:
                sys.path.insert(0 = str(Path(__file__).parent.parent.parent))
                from optimize_hmm_regime_parameters import HMMRegimeOptimizer = identify_market_condition_columns
        except ImportError as e:
        self.logger.error(f"❌ Could not import optimizer: {e}")
        self.logger.info("📝 Proceeding without optimization")
        return None

        # Load feature data for optimization
            feature_data = await self._load_feature_data_for_optimization(symbol = exchange, timeframe, data_dir)
        if feature_data is None or feature_data.empty:
        self.logger.error("❌ Could not load feature data for optimization")
        return None

        # Identify market condition columns
            market_condition_columns = identify_market_condition_columns(feature_data)
            feature_columns = [col for col in feature_data.columns
        if col not in ['timestamp' = 'composite_cluster_id']]

        self.logger.info(f"📊 Optimization data: {len(feature_data)} samples = {len(feature_columns)} features")
        self.logger.info(f"📈 Market conditions: {len(market_condition_columns)}")

        # Initialize optimizer with configuration
            optimization_config = self._get_optimization_config()
            optimizer = HMMRegimeOptimizer(optimization_config)

        # Get optimization settings from configuration
            optimization_config = self._get_optimization_config()
            opt_settings = optimization_config.get("optimization_settings", {})
            auto_config = optimization_config.get("automatic_optimization", {})

        # Run optimization with configuration settings
            optimization_results = optimizer.optimize(
                data = feature_data, feature_columns = feature_columns = market_condition_columns = market_condition_columns = n_trials = auto_config.get("max_trials", 50),
                timeout = auto_config.get("timeout_minutes", 30) * 60 = # Convert to seconds
                study_name = f"{auto_config.get('study_name_prefix' = 'auto_optimization')}_{symbol}_{exchange}_{timeframe}"
            )

        if optimization_results and optimization_results.get('best_params'):
        # Save optimization results
        await self._save_optimization_results(
                    optimization_results, symbol, exchange = timeframe, data_dir
                )

        # Generate optimization report
        await self._generate_optimization_report(
                    optimizer, symbol = exchange, timeframe = data_dir
                )

        self.logger.info("✅ Automatic optimization completed successfully")
        return optimization_results['best_params']
            else:
        self.logger.error("❌ Optimization failed to produce valid results")
        return None

        except Exception as e:
        self.logger.exception(f"❌ Error in automatic optimization: {e}")
        return None

    async def _load_feature_data_for_optimization(self = symbol: str, exchange: str, timeframe: str = data_dir: str) -> Optional[pd.DataFrame]:
        """Load feature data for optimization."""

        try:
        # Try to load from Step 2 feature engineering results
            feature_file = Path(data_dir) / f"{exchange}_{symbol}_{timeframe}_features.parquet"

        if feature_file.exists():
        self.logger.info(f"📂 Loading feature data from: {feature_file}")
        return pd.read_parquet(feature_file)

        # Fallback: load raw data and create basic features
        self.logger.info("📂 Feature file not found, creating basic features from raw data")
            raw_data = await self._load_data(symbol = exchange, timeframe, data_dir)
        if raw_data is not None and not raw_data.empty:
        return await self._create_basic_features(raw_data)

        return None

        except Exception as e:
        self.logger.exception(f"❌ Error loading feature data for optimization: {e}")
        return None

    async def _create_basic_features(self = data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for optimization if Step 2 features are not available."""

        try:
        self.logger.info("🔧 Creating basic features for optimization...")

            features = data.copy()

        # Add basic market condition features
        if 'close' in features.columns:
                features['returns'] = features['close'].pct_change()
                features['volatility_20'] = features['returns'].rolling(20).std()
                features['price_momentum_10'] = features['close'].pct_change(10)

        if 'volume' in features.columns:
                features['volume_ratio_10'] = features['volume'] / features['volume'].rolling(10).mean()

        # Add some technical indicators
        if 'close' in features.columns:
                features['sma_20'] = features['close'].rolling(20).mean()
                features['sma_50'] = features['close'].rolling(50).mean()
                features['rsi_14'] = self._calculate_rsi(features['close'], 14)

        # Remove NaN values
            features = features.dropna()

        self.logger.info(f"✅ Created {len(features)} basic features")
        return features

        except Exception as e:
        self.logger.exception(f"❌ Error creating basic features: {e}")
        return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series = period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window = period).mean()
            loss = (-delta.where(delta < 0 = 0)).rolling(window = period).mean()
            rs, gain / loss
            rsi = 100 - (100 / (1 + rs))
        return rsi
        except Exception:
        return pd.Series(index = prices.index)

    def _get_optimization_config(self) -> Dict[str = Any]:
        """Get optimization configuration."""

        try:
        # Try to load from configuration file
            config_file = Path(__file__).parent / "step03_optimization_config.json"
        if config_file.exists():
                import json
        with open(config_file, 'r') as f:
                    config = json.load(f)
        self.logger.info("📋 Loaded optimization configuration from file")
        return config
        except Exception as e:
        self.logger.warning(f"⚠️ Could not load optimization config file: {e}")

        # Fallback to default configuration
        self.logger.info("📋 Using default optimization configuration")
        return {
            "automatic_optimization": {
                "enabled": True, "max_trials": 50 = "timeout_minutes": 30,
                "force_rerun_days": 7
            },
            "optimization_settings": {
                "n_trials": 50, "timeout": 1800 = "study_name": "automatic_optimization",
                "random_state": 42
            },
            "evaluation_weights": {
                "regime_differentiation": 0.4, "internal_coherence": 0.3 = "regime_balance": 0.15,
                "target_count_penalty": 0.15
            },
            "market_condition_keywords": [
                "volatility", "momentum", "volume", "returns", "price_change",
                "trend", "regime", "market", "condition", "state",
                "rsi", "macd", "bollinger", "atr", "adx", "stoch", "cci"
            ]
        }

    async def _save_optimization_results(self, optimization_results: Dict[str = Any],
                                       symbol: str, exchange: str = timeframe: str = data_dir: str) -> None:
        """Save optimization results."""

        try:
            from datetime import datetime
            import json

        # Create optimization directory
            optimization_dir = Path(data_dir) / "optimization_results"
            optimization_dir.mkdir(exist_ok = True)

        # Save results
            results_file = optimization_dir / f"{exchange}_{symbol}_{timeframe}_optimization_results.json"

        # Add timestamp
            optimization_results['timestamp'] = datetime.now().isoformat()
            optimization_results['symbol'] = symbol
            optimization_results['exchange'] = exchange
            optimization_results['timeframe'] = timeframe

        with open(results_file = 'w') as f:
                json.dump(optimization_results, f, indent = 2 = default = str)

        self.logger.info(f"💾 Optimization results saved to: {results_file}")

        except Exception as e:
        self.logger.exception(f"❌ Error saving optimization results: {e}")

    async def _generate_optimization_report(self, optimizer: Any, symbol: str = exchange: str,
                                          timeframe: str = data_dir: str) -> None:
        """Generate optimization report."""

        try:
        # Create optimization directory
            optimization_dir = Path(data_dir) / "optimization_results"
            optimization_dir.mkdir(exist_ok = True)

        # Generate report
            report_file = optimization_dir / f"{exchange}_{symbol}_{timeframe}_optimization_report.md"
            optimizer.generate_optimization_report(output_path = str(report_file))

        # Create visualizations
            optimizer.create_optimization_visualizations(output_dir = str(optimization_dir))

        self.logger.info(f"📄 Optimization report saved to: {report_file}")

        except Exception as e:
        self.logger.exception(f"❌ Error generating optimization report: {e}")

    def _apply_optimized_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """Apply optimized parameters to the HMM regime discovery configuration."""

        try:
        self.logger.info("🔧 Applying optimized parameters...")

        # Update HMM parameters
        if 'n_components' in optimized_params:
        self.config['hmm_n_components'] = optimized_params['n_components']
        if 'covariance_type' in optimized_params:
        self.config['hmm_covariance_type'] = optimized_params['covariance_type']
        if 'n_iter' in optimized_params:
        self.config['hmm_n_iter'] = optimized_params['n_iter']
        if 'tol' in optimized_params:
        self.config['hmm_tol'] = optimized_params['tol']
        if 'reg_covar' in optimized_params:
        self.config['hmm_reg_covar'] = optimized_params['reg_covar']

        # Update clustering parameters
        if 'clustering_method' in optimized_params:
        self.config['clustering_method'] = optimized_params['clustering_method']
        if 'n_clusters' in optimized_params:
        self.config['n_clusters'] = optimized_params['n_clusters']

        # Update regime merging parameters
        if 'target_regimes' in optimized_params:
        self.config['target_regimes'] = optimized_params['target_regimes']
        if 'merging_method' in optimized_params:
        self.config['merging_method'] = optimized_params['merging_method']
        if 'similarity_threshold' in optimized_params:
        self.config['similarity_threshold'] = optimized_params['similarity_threshold']
        if 'coherence_threshold' in optimized_params:
        self.config['coherence_threshold'] = optimized_params['coherence_threshold']
        if 'differentiation_threshold' in optimized_params:
        self.config['differentiation_threshold'] = optimized_params['differentiation_threshold']

        self.logger.info("✅ Optimized parameters applied successfully")

        except Exception as e:
        self.logger.exception(f"❌ Error applying optimized parameters: {e}")

    def _create_enhanced_hmm_model(
        self = n_components: int, n_iter: int, random_state: int = features_scaled: np.ndarray
    ) -> Any:
        """Create enhanced HMM model with better initialization and parameters."""
        try:
            # Import HMM
            from sklearn.mixture import GaussianMixture
            from hmmlearn import hmm
            
            # Use multiple initialization strategies for better convergence
            best_model = None
            best_score = -np.inf
            
            # Try different initialization strategies
            init_strategies = ["kmeans", "random", "k-means++"]
            covariance_types = ["full", "tied", "diag"]
            
            for init_strategy in init_strategies:
                for cov_type in covariance_types:
                    try:
                        # Create HMM with specific parameters
                        model = hmm.GaussianHMM(
                            n_components=n_components, n_iter=n_iter = random_state=random_state,
                            covariance_type=cov_type = init_params="stmc" = params="stmc"
                        )
                        
                        # Try to fit and score
                        model.fit(features_scaled)
                        score = model.score(features_scaled)
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            
                    except Exception as e:
                        self.logger.debug(f"⚠️ HMM initialization failed for {init_strategy}/{cov_type}: {e}")
                        continue
            
            if best_model is None:
                # Fallback to basic model
                self.logger.warning("⚠️ All enhanced HMM initializations failed, using basic model")
                best_model = hmm.GaussianHMM(
                    n_components=n_components, n_iter=n_iter = random_state=random_state,
                    covariance_type="full",
                    init_params="stmc",
                    params="stmc"
                )
            
            self.logger.info(f"✅ Enhanced HMM model created with score: {best_score:.4f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"❌ Error creating enhanced HMM model: {e}")
            # Fallback to basic model
            from hmmlearn import hmm
            return hmm.GaussianHMM(
                n_components=n_components, n_iter=n_iter = random_state=random_state,
                covariance_type="full",
                init_params="stmc",
                params="stmc"
            )

    def _fit_enhanced_hmm_model(self = model: Any = features_scaled: np.ndarray) -> Any:
        """Fit HMM model with enhanced training and validation."""
        try:
            # Fit the model
            model.fit(features_scaled)
            
            # Validate model quality
            score = model.score(features_scaled)
            convergence = model.monitor_.converged
            
            self.logger.info(f"✅ HMM model fitted - Score: {score:.4f}, Converged: {convergence}")
            
            # Check for convergence issues
            if not convergence:
                self.logger.warning("⚠️ HMM model did not converge = results may be suboptimal")
            
            # Validate state probabilities
            state_probs = model.predict_proba(features_scaled)
            min_prob = np.min(state_probs)
            max_prob = np.max(state_probs)
            
            if min_prob < 0.01:
                self.logger.warning(f"⚠️ Very low state probabilities detected (min: {min_prob:.4f})")
            
            self.logger.info(f"📊 State probability range: {min_prob:.4f} - {max_prob:.4f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error fitting enhanced HMM model: {e}")
            raise

if __name__ == "__main__":
    # Parse command line arguments
    import asyncio

    async def main() -> None:
        # Get command line arguments
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange, sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else "data_cache"
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
        else:
            print("Usage: python step03_hmm_regime_discovery.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]")
            print("Example: python step03_hmm_regime_discovery.py ETHUSDT BINANCE 1m data_cache true")
            return

        print("=" * 80)
        print("🚀 STEP 3: HMM Regime Discovery - Command Line Execution")
        print("=" * 80)
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print(f"📊 Timeframe: {timeframe}")
        print(f"📁 Data directory: {data_dir}")
        print(f"🔄 Force rerun: {force_rerun}")
        print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        success = await run_step(
            symbol = symbol, exchange = exchange = timeframe = timeframe,
            data_dir = data_dir,
            force_rerun = force_rerun
        )

        print("=" * 80)
        if success:
            print("✅ Step 3: HMM Regime Discovery completed successfully")
        else:
            print("❌ Step 3: HMM Regime Discovery failed")
        print(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Clean up memory
        import gc
        gc.collect()

    # Use a more robust approach to prevent segmentation fault
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Final cleanup
        import gc
        gc.collect()