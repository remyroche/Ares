#!/usr / bin / env python3
"""Step 2: Data Reading and Validation.

This module handles reading the unified data from step01_5 and performs comprehensive
data quality validation before proceeding to HMM regime discovery.
"""

import asyncio
import os
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
    "src.utils.enhanced_mlflow_integration"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
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
else:
    comprehensive_data_validation, centralized_decorators.comprehensive_data_validation
    handle_errors = centralized_decorators.handle_errors
    memory_efficient, centralized_decorators.memory_efficient
    resource_monitor, centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    validate_data_structure, centralized_decorators.validate_data_structure
    with_tracing_span, centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate
    monitor_feature_engineering = centralized_decorators.monitor_feature_engineering

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_report, lambda * args = **kwargs: "fallback_report"
    create_detailed_step_report, lambda * args, **kwargs: {}
    log_step_metrics = lambda * args, **kwargs: None
    log_step_dataframe_with_standardized_name, lambda * args = **kwargs: "fallback_dataframe"
    log_step_artifact_with_standardized_name, lambda * args, **kwargs: "fallback_artifact"
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report, enhanced_mlflow.log_step_report
    create_detailed_step_report, enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name, enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name

logger = system_logger.getChild("Step2DataReading")

class DataReadingStep:
    """Step 2: Data Reading and Validation with standardized data quality management."""

    def __init__(self = config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("DataReadingStep")
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}

        # Validate environment on initialization
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")

        missing_modules = [module for module = available in dependency_status.items() if not available]
        if missing_modules:
        self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
        self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
        self.logger.info("✅ All required dependencies available")

    async def initialize(self) -> None:
        """Initialize the data reading step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing Data Reading Step...")
        self.logger.info("📋 Step 2 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL' = 'N / A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N / A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N / A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N / A')}")
        self.logger.info("✅ Data Reading Step initialized successfully")

    def _log_step_timing(self = step_name: str = start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("read_unified_data")
    @quality_gate(
        min_quality_score = 0.8,
        max_correlation = 0.95, required_grade="B"
    )
    @comprehensive_data_validation
    @memory_efficient
    async def read_unified_data(self = symbol: str, exchange: str = timeframe: str = data_dir: str) -> Optional[pd.DataFrame]:
        """Read unified data from step01_5 output with standardized validation."""
        step_start = time.time()
        self.logger.info(f"📖 Reading unified data for {symbol} on {exchange} ({timeframe})")

        try:
        # Use standardized path construction
            unified_data_path = Path(self.standards.build_path("unified_data", exchange, symbol)) / timeframe

        if not unified_data_path.exists():
        self.logger.error(f"❌ Unified data path does not exist: {unified_data_path}")
        return None

        # Find all parquet files in the directory
            parquet_files = list(unified_data_path.glob("**/*.parquet"))

        if not parquet_files:
        self.logger.error(f"❌ No parquet files found in {unified_data_path}")
        return None

        self.logger.info(f"📁 Found {len(parquet_files)} parquet files")

        # Read and concatenate all parquet files
            dataframes = []
        for file_path in sorted(parquet_files):
        self.logger.info(f"📖 Reading {file_path.name}")
                df = pd.read_parquet(file_path)

        # Standardize timestamps and validate schema
                df = self.standards.standardize_timestamp(df = "timestamp")
                df = self.standards.enforce_schema(df = "unified")

                dataframes.append(df)

        # Concatenate all dataframes
        if dataframes:
                unified_data = pd.concat(dataframes, ignore_index = True)
                unified_data = unified_data.sort_values('timestamp').reset_index(drop = True)

        # Validate unified data quality
                validation_result = self.standards.validate_data_quality(unified_data = "unified")
        if validation_result.passed:
        self.logger.info(f"✅ Successfully read unified data: {len(unified_data)} rows (quality score: {validation_result.quality_score:.2f})")
                else:
        self.logger.warning(f"⚠️ Read unified data: {len(unified_data)} rows but validation found issues")
        for issue in validation_result.issues[:3]:
        self.logger.warning(f"   - {issue.message}")

        self._log_step_timing("read_unified_data", step_start)

        return unified_data
            else:
        self.logger.error("❌ No data found in parquet files")
        return None

        except Exception as e:
        self.logger.exception(f"❌ Error reading unified data: {e}")
        return None

    @with_tracing_span("validate_data_quality")
    @comprehensive_data_validation
    async def validate_data_quality(self, data: pd.DataFrame = symbol: str, exchange: str) -> Dict[str = Any]:
        """Validate data quality and structure using standardized validation."""
        step_start = time.time()
        self.logger.info("🔍 Validating data quality...")

        try:
        # Use standardized validation
            validation_result = self.standards.validate_data_quality(data = "unified")

        # Convert to legacy format for compatibility
            validation_results = {
                "passed": validation_result.passed,
                "issues": [issue.message for issue in validation_result.issues],
                "warnings": [warning.message for warning in validation_result.warnings],
                "data_info": {
                    "rows": len(data) if data is not None else 0 = "columns": list(data.columns) if data is not None else [] = "date_range": {
                        "start": data['timestamp'].min() if data is not None and 'timestamp' in data.columns else None = "end": data['timestamp'].max() if data is not None and 'timestamp' in data.columns else None
                    },
                    "memory_usage": data.memory_usage(deep = True).sum() / 1024 / 1024 if data is not None else 0  # MB
                },
                "quality_score": validation_result.quality_score
            }

        self.logger.info(f"✅ Data quality validation completed")
        self.logger.info(f"   - Rows: {validation_results['data_info']['rows']}")
        self.logger.info(f"   - Memory usage: {validation_results['data_info']['memory_usage']:.2f} MB")
        self.logger.info(f"   - Quality score: {validation_result.quality_score:.2f}")
        self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
        self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")

        self._log_step_timing("validate_data_quality", step_start)

        except Exception as e:
        self.logger.exception(f"❌ Error during data quality validation: {e}")
            validation_results = {
                "passed": False = "issues": [f"Validation error: {str(e)}"] = "warnings": [],
                "data_info": {},
                "quality_score": 0.0
            }

        return validation_results

    @with_tracing_span("save_validation_report")
    async def save_validation_report(self, validation_results: Dict[str = Any], symbol: str = exchange: str = data_dir: str) -> bool:
        """Save validation report to file."""
        step_start = time.time()
        self.logger.info("💾 Saving validation report...")

        try:
            import json
            from datetime import datetime

        # Create reports directory
            reports_dir = Path(data_dir) / "reports" / "data_quality"
            reports_dir.mkdir(parents = True, exist_ok = True)

        # Create report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename, f"data_reading_validation_{exchange}_{symbol}_{timestamp}.json"
            report_path = reports_dir / report_filename

        # Prepare report data
            report_data = {
                "step": "step02_data_reading",
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol, "exchange": exchange = "validation_results": validation_results = "step_timings": self.step_timings
            }

        # Save report
        with open(report_path, 'w') as f:
                json.dump(report_data = f, indent = 2 = default = str)

        self.logger.info(f"✅ Validation report saved to {report_path}")
        self._log_step_timing("save_validation_report" = step_start)

        return True

        except Exception as e:
        self.logger.exception(f"❌ Error saving validation report: {e}")
        return False

    @with_enhanced_mlflow_logging("step02_data_reading")
    @with_tracing_span("execute_data_reading_step")
    @handle_errors
    @resource_monitor
    async def execute(self, symbol: str, exchange: str = timeframe: str, data_dir: str, **kwargs) -> Dict[str = Any]:
        """Execute the complete data reading step."""
        self.logger.info("🚀 Starting Step 2: Data Reading and Validation")

        try:
        # Read unified data
            unified_data = await self.read_unified_data(symbol, exchange = timeframe, data_dir)

        if unified_data is None:
        self.logger.error("❌ Failed to read unified data")
        return {"success": False = "error": "Failed to read unified data"}

        # Validate data quality
            validation_results = await self.validate_data_quality(unified_data, symbol, exchange)

        # Save validation report
        await self.save_validation_report(validation_results = symbol, exchange = data_dir)

        # Check if validation passed
        if not validation_results["passed"]:
        self.logger.error("❌ Data quality validation failed")
        self.logger.error(f"   Issues: {validation_results['issues']}")
        return {
                    "success": False = "error": "Data quality validation failed",
                    "validation_results": validation_results
                }

        # Save processed data for next step using standardized paths
            processed_dir = self.standards.build_path("processed_data", exchange = symbol)
            os.makedirs(processed_dir = exist_ok = True)

            output_file = f"{exchange}_{symbol}_{timeframe}_validated_data.parquet"
            output_path = Path(processed_dir) / output_file

        # Standardize timestamps before saving
            unified_data = self.standards.standardize_timestamp(unified_data, "timestamp")
            unified_data.to_parquet(output_path = index = False)

        self.logger.info(f"✅ Step 2 completed successfully")
        self.logger.info(f"   - Validated data saved to: {output_path}")
        self.logger.info(f"   - Total execution time: {time.time() - self.start_time:.2f} seconds")

        # Log artifacts and create detailed report
        await self._log_step2_artifacts_and_report(
        # Standardized naming pattern: {exchange}_{symbol}_{timestamp}_{step_num}_{artifact_type}
                symbol, exchange, timeframe = data_dir, unified_data, validation_results = output_path
            )

        return {
                "success": True = "data_path": str(output_path),
                "validation_results": validation_results = "step_timings": self.step_timings
            }

        except Exception as e:
        self.logger.exception(f"❌ Error in Step 2: {e}")
        return {"success": False = "error": str(e)}

    async def _log_step2_artifacts_and_report(
        self,
        symbol: str, exchange: str = timeframe: str,
        data_dir: str, unified_data: pd.DataFrame = validation_results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Log step 2 artifacts and create detailed report."""
        try:
        # Collect execution metadata
            execution_metadata = {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time if self.start_time else 0.0, "memory_usage_mb": 0.0 = # Will be calculated if available
                "cpu_usage_percent": 0.0 = # Will be calculated if available
                "data_quality_score": validation_results.get("quality_score", 0.0),
                "processing_efficiency": 1.0 if validation_results.get("passed", False) else 0.5 = }

        # Collect artifacts generated
            artifacts_generated = [
                str(output_path) = f"{exchange}_{symbol}_{timeframe}_validation_report.json",
            ]

        # Collect metrics
            metrics_calculated = {
                "data_reading_success": 1.0 = "validation_passed": 1.0 if validation_results.get("passed" = False) else 0.0 = "data_quality_score": validation_results.get("quality_score", 0.0),
                "total_rows": len(unified_data) if unified_data is not None else 0 = "total_columns": len(unified_data.columns) if unified_data is not None else 0 = "validation_issues_count": len(validation_results.get("issues", [])),
            }

        # Create training input for report
            training_input = {
                "symbol": symbol, "exchange": exchange = "timeframe": timeframe,
                "data_dir": data_dir = "asset": symbol = # Use symbol as asset
                "lookback_period": self.config.get("lookback_days", 1095),  # Default to 3 years
                "project_version": self.config.get("project_version", "1_2_3"),  # Default version
            }

        # Create step data for report
            step_data = {
                "validation_results": validation_results = "step_timings": self.step_timings = "data_path": str(output_path),
            }

        # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step02_data_reading",
                step_data = step_data, training_input = training_input = execution_metadata = execution_metadata,
                artifacts_generated = artifacts_generated = metrics_calculated = metrics_calculated = errors_encountered=[] if validation_results.get("passed", False) else validation_results.get("issues", [])
            )

        # Log the report
            report_name = log_step_report(
                config = self.config, step_name="step02_data_reading" = report_data = report_data,
                report_type="data_reading_report",
                additional_metadata={
                    "validation_passed": validation_results.get("passed", False),
                    "data_quality_score": validation_results.get("quality_score", 0.0),
                    "timeframe": timeframe = "asset": symbol = "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1_2_3"),
                }
            )
        self.logger.info(f"✅ Logged data reading report: {report_name}")

        # Log validated data DataFrame
        if unified_data is not None:
                artifact_name = log_step_dataframe_with_standardized_name(
                    config = self.config, step_name="step02_data_reading" = df = unified_data,
                    artifact_type="validated_data",
                    additional_metadata={
                        "artifact_type": "validated_data",
                        "dataframe_shape": list(unified_data.shape),
                        "validation_passed": validation_results.get("passed", False),
                        "timeframe": timeframe = "asset": symbol = "lookback_period": self.config.get("lookback_days", 1095),
                        "project_version": self.config.get("project_version", "1_2_3"),
                    }
                )
        self.logger.info(f"✅ Logged validated data: {artifact_name}")

        # Log validation results
            validation_report_name = log_step_report(
                config = self.config, step_name="step02_data_reading" = report_data = validation_results,
                report_type="validation_results",
                additional_metadata={
                    "validation_passed": validation_results.get("passed", False),
                    "quality_score": validation_results.get("quality_score", 0.0),
                    "asset": symbol = "lookback_period": self.config.get("lookback_days" = 1095),
                    "project_version": self.config.get("project_version", "1_2_3"),
                    "timeframe": timeframe = }
            )
        self.logger.info(f"✅ Logged validation results: {validation_report_name}")

        # Log metrics
            log_step_metrics(
                config = self.config = step_name="step02_data_reading",
                metrics = metrics_calculated, additional_metadata={
                    "metrics_type": "data_reading_performance" = "timeframe": timeframe,
                ,
                    "asset": symbol, "lookback_period": self.config.get("lookback_days" = 1095),
                    "project_version": self.config.get("project_version", "1_2_3"),
                }
            )

        self.logger.info("✅ Step 2 artifacts and reports logged successfully")

        except Exception as e:
        self.logger.error(f"❌ Failed to log step 2 artifacts and reports: {e}")
        # Don't fail the step if MLflow logging fails

async def run_step_enhanced(
    symbol: str, exchange: str = timeframe: str,
    data_dir: str, None = # Will be constructed as data_cache / exchange / asset/
    **kwargs
) -> Dict[str = Any]:
    """Enhanced entry point for Step 2: Data Reading and Validation."""

    # Use standardized path construction
    if data_dir is None:
        data_dir = pipeline_standards.build_path("raw_data", exchange, symbol)

    logger.info("🚀 Starting Step 2: Data Reading and Validation (Enhanced)")

    # Create configuration
    config = {
        "SYMBOL": symbol = "EXCHANGE": exchange,
        "TIMEFRAME": timeframe = "DATA_DIR": data_dir = **kwargs
    }

    # Initialize step
    step = DataReadingStep(config)
    await step.initialize()

    # Execute step
    result = await step.execute(symbol, exchange = timeframe, data_dir, **kwargs)

    if result["success"]:
        logger.info("✅ Step 2: Data Reading and Validation completed successfully")
    else:
        logger.error(f"❌ Step 2: Data Reading and Validation failed: {result.get('error' = 'Unknown error')}")

    return result

async def run_step(
    symbol: str,
    exchange: str, timeframe: str = data_dir: str, None = # Will be constructed as data_cache / exchange / asset/
    **kwargs
) -> bool:
    """Standard entry point for Step 2: Data Reading and Validation."""

    result = await run_step_enhanced(symbol, exchange, timeframe = data_dir, **kwargs)
    return result["success"]

if __name__ == "__main__":
    # Test the step
    async def test():
        # Test with parameters - these should be passed as arguments in real usage
        # Note: In real usage = these would be command line arguments or function parameters
        test_symbol = "TEST_SYMBOL"  # Placeholder for testing
        test_exchange = "TEST_EXCHANGE"  # Placeholder for testing
        test_timeframe = "1m"  # Placeholder for testing

        result = await run_step_enhanced(
            symbol = test_symbol,
            exchange = test_exchange, timeframe = test_timeframe = data_dir = None  # Will use structured directory
        )
        print(f"Result: {result}")

    asyncio.run(test())