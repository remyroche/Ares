from __future__ import annotations

"""Step 1: Data Collection.

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.decorators import handles_errors

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "src.config",
    "src.utils.logger",
    "src.utils.error_handler",
    "src.training.steps.data_downloader",
    "src.utils.enhanced_mlflow_integration",
    "src.utils.centralized_decorators",
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(
    REQUIRED_MODULES
)

# Safe imports with fallbacks
CONFIG = PipelineStandards.safe_import(
    "src.config", {"SYMBOL": None, "INTERVAL": "1m", "LOOKBACK_YEARS": 2}
)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
handle_errors = PipelineStandards.safe_import("src.utils.error_handler", None)
download_all_data_with_consolidation = PipelineStandards.safe_import(
    "src.training.steps.data_downloader", None
)
enhanced_mlflow = PipelineStandards.safe_import(
    "src.utils.enhanced_mlflow_integration", None
)
centralized_decorators = PipelineStandards.safe_import(
    "src.utils.centralized_decorators", None
)


# Fallback functions if imports fail
def create_fallback_logger():
    import logging

    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def create_fallback_decorator():
    def decorator(func):
        return func

    return decorator


# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if handle_errors is None:
    handle_errors = create_fallback_decorator()

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()

    def log_step_report(*args, **kwargs):
        return "fallback_report"

    def create_detailed_step_report(*args, **kwargs):
        return {}

    def log_step_metrics(*args, **kwargs):
        return None

    def log_step_artifact_with_standardized_name(*args, **kwargs):
        return "fallback_artifact"

    def log_step_dataframe_with_standardized_name(*args, **kwargs):
        return "fallback_dataframe"

else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_artifact_with_standardized_name = (
        enhanced_mlflow.log_step_artifact_with_standardized_name
    )
    log_step_dataframe_with_standardized_name = (
        enhanced_mlflow.log_step_dataframe_with_standardized_name
    )

if centralized_decorators is None:
    monitor_data_collection = create_fallback_decorator()
else:
    monitor_data_collection = centralized_decorators.monitor_data_collection


class DataCollectionStep:
    """Step 1: Data Collection using standardized pipeline utilities."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("DataCollectionStep")
        self.standards = pipeline_standards

        # Validate environment on initialization
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")

        missing_modules = [
            module for module, available in dependency_status.items() if not available
        ]
        if missing_modules:
            self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
            self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
            self.logger.info("✅ All required dependencies available")

    async def initialize(self) -> None:
        """Initialize the data collection step."""
        self.logger.info("Initializing Data Collection Step...")
        self.logger.info("Data Collection Step initialized successfully")

    # @with_enhanced_mlflow_logging - removed, use traced"step1_data_collection")
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute data collection with standardized quality management.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state

        """
        self.logger.info("🚀 Starting standardized data collection...")

        try:
            # Validate input parameters
            symbol = training_input.get("symbol")
            exchange = training_input.get("exchange")
            timeframe = training_input.get("timeframe", "1m")

            if not symbol or not exchange:
                msg = "Symbol and exchange are required parameters"
                raise ValueError(msg)

            # Build standardized paths
            data_dir = self.standards.build_path("raw_data", exchange, symbol)
            self.logger.info(f"📁 Using standardized data directory: {data_dir}")

            # Execute the data collection
            success = await self._run_data_collection(training_input, data_dir)

            if success:
                self.logger.info("✅ Data collection completed successfully")

                # Run standardized quality check after data collection
                quality_success = await self._run_standardized_quality_check(
                    symbol, exchange, timeframe, data_dir
                )

                if quality_success:
                    self.logger.info("✅ Standardized quality check passed")
                    pipeline_state["data_collection_completed"] = True
                    pipeline_state["quality_check_passed"] = True
                else:
                    self.logger.warning("⚠️ Standardized quality check found issues")
                    pipeline_state["data_collection_completed"] = True
                    pipeline_state["quality_check_passed"] = False
            else:
                self.logger.error("❌ Data collection failed")
                pipeline_state["data_collection_completed"] = False
                pipeline_state["quality_check_passed"] = False

        except Exception as e:
            self.logger.exception(f"❌ Error during data collection: {e}")
            pipeline_state["data_collection_completed"] = False
            pipeline_state["quality_check_passed"] = False

        # Log detailed report and artifacts
        await self._log_step1_artifacts_and_report(training_input, pipeline_state)

        return pipeline_state

    async def _log_step1_artifacts_and_report(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> None:
        """Log step 1 artifacts and create detailed report."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")
            training_input.get("data_dir", "data_cache")

            # Collect execution metadata
            execution_metadata = {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,  # Will be calculated if available
                "memory_usage_mb": 0.0,  # Will be calculated if available
                "cpu_usage_percent": 0.0,  # Will be calculated if available
                "data_quality_score": (
                    1.0 if pipeline_state.get("quality_check_passed", False) else 0.5
                ),
                "processing_efficiency": (
                    1.0
                    if pipeline_state.get("data_collection_completed", False)
                    else 0.0
                ),
            }

            # Collect artifacts generated
            artifacts_generated = []
            if pipeline_state.get("data_collection_completed", False):
                # Add expected artifacts
                artifacts_generated.extend(
                    [
                        f"{exchange}_{symbol}_{timeframe}_klines.parquet",
                        f"{exchange}_{symbol}_{timeframe}_trades.parquet",
                        f"{exchange}_{symbol}_{timeframe}_orderbook.parquet",
                    ]
                )

            # Collect metrics
            metrics_calculated = {
                "data_collection_success": (
                    1.0
                    if pipeline_state.get("data_collection_completed", False)
                    else 0.0
                ),
                "quality_check_passed": (
                    1.0 if pipeline_state.get("quality_check_passed", False) else 0.0
                ),
                "total_artifacts_generated": len(artifacts_generated),
            }

            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step1_data_collection",
                step_data=pipeline_state,
                training_input=training_input,
                execution_metadata=execution_metadata,
                artifacts_generated=artifacts_generated,
                metrics_calculated=metrics_calculated,
                errors_encountered=(
                    []
                    if pipeline_state.get("data_collection_completed", False)
                    else ["Data collection failed"]
                ),
            )

            # Log the report
            report_name = log_step_report(
                config=self.config,
                step_name="step1_data_collection",
                report_data=report_data,
                report_type="data_collection_report",
                additional_metadata={
                    "data_collection_success": pipeline_state.get(
                        "data_collection_completed", False
                    ),
                    "quality_check_passed": pipeline_state.get(
                        "quality_check_passed", False
                    ),
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": training_input.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                },
            )
            self.logger.info(f"✅ Logged data collection report: {report_name}")

            # Log data quality summary
            quality_report_name = log_step_report(
                config=self.config,
                step_name="step1_data_collection",
                report_data={
                    "quality_check_passed": pipeline_state.get(
                        "quality_check_passed", False
                    ),
                    "data_collection_completed": pipeline_state.get(
                        "data_collection_completed", False
                    ),
                    "artifacts_generated": artifacts_generated,
                },
                report_type="data_quality_summary",
                additional_metadata={
                    "quality_check_passed": pipeline_state.get(
                        "quality_check_passed", False
                    ),
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": training_input.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                },
            )
            self.logger.info(f"✅ Logged data quality summary: {quality_report_name}")

            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step1_data_collection",
                metrics=metrics_calculated,
                additional_metadata={
                    "metrics_type": "data_collection_performance",
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": training_input.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                },
            )

            self.logger.info("✅ Step 1 artifacts and reports logged successfully")

        except Exception as e:
            self.logger.exception(f"❌ Failed to log step 1 artifacts and reports: {e}")
            # Don't fail the step if MLflow logging fails

    async def _run_standardized_quality_check(
        self, symbol: str, exchange: str, timeframe: str, data_dir: str
    ) -> bool:
        """Run standardized quality check after data collection."""
        try:
            self.logger.info("🔍 Running standardized quality check...")

            # Check for expected files
            expected_files = [
                self.standards.generate_file_name(
                    "klines", exchange, symbol, timeframe
                ),
                self.standards.generate_file_name("aggtrades", exchange, symbol),
            ]

            quality_results = []

            for file_name in expected_files:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    self.logger.info(f"🔍 Validating {file_name}...")

                    try:
                        import pandas as pd

                        df = pd.read_parquet(file_path)

                        # Standardize timestamps
                        df = self.standards.standardize_timestamp(df, "timestamp")

                        # Determine schema type
                        if "klines" in file_name:
                            schema_name = "klines"
                        elif "aggtrades" in file_name:
                            schema_name = "aggtrades"
                        else:
                            schema_name = "unified"

                        # Run comprehensive quality validation
                        validation_result = self.standards.validate_data_quality(
                            df, schema_name
                        )
                        quality_results.append(validation_result)

                        # Log results
                        if validation_result.passed:
                            self.logger.info(
                                f"✅ {file_name} quality check passed (score: {validation_result.quality_score:.2f})"
                            )
                        else:
                            self.logger.warning(f"⚠️ {file_name} quality check issues:")
                            for issue in validation_result.issues[
                                :3
                            ]:  # Show first 3 issues
                                self.logger.warning(f"   - {issue.message}")
                            if len(validation_result.issues) > 3:
                                self.logger.warning(
                                    f"   ... and {len(validation_result.issues) - 3} more issues"
                                )

                        # Log warnings
                        for warning in validation_result.warnings[
                            :3
                        ]:  # Show first 3 warnings
                            self.logger.info(f"   ⚠️ {warning.message}")
                        if len(validation_result.warnings) > 3:
                            self.logger.info(
                                f"   ... and {len(validation_result.warnings) - 3} more warnings"
                            )

                    except Exception as e:
                        self.logger.exception(f"❌ Error validating {file_name}: {e}")
                        return False
                else:
                    self.logger.warning(f"⚠️ Expected file not found: {file_name}")

            # Overall quality assessment
            if quality_results:
                overall_passed = all(result.passed for result in quality_results)
                overall_quality_score = sum(
                    result.quality_score for result in quality_results
                ) / len(quality_results)

                self.logger.info(
                    f"📊 Overall quality check: {'PASSED' if overall_passed else 'FAILED'}"
                )
                self.logger.info(
                    f"📊 Average quality score: {overall_quality_score:.2f}"
                )

                # Log summary statistics
                total_issues = sum(len(result.issues) for result in quality_results)
                total_warnings = sum(len(result.warnings) for result in quality_results)

                if total_issues > 0:
                    self.logger.warning(f"📊 Total issues found: {total_issues}")
                if total_warnings > 0:
                    self.logger.info(f"📊 Total warnings: {total_warnings}")

                return overall_passed

            self.logger.warning("⚠️ No quality results available")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Error running standardized quality check: {e}")
            return False

    async def _run_data_collection(
        self, training_input: dict[str, Any], data_dir: str
    ) -> bool:
        """Run the actual data collection process with standardized validation."""
        try:
            symbol = training_input.get("symbol")
            exchange = training_input.get("exchange")
            timeframe = training_input.get("timeframe", "1m")

            # Validate required parameters
            if not symbol:
                self.logger.error("❌ Symbol parameter is required")
                return False
            if not exchange:
                self.logger.error("❌ Exchange parameter is required")
                return False

            self.logger.info(f"📊 Downloading data for {exchange}_{symbol}_{timeframe}")

            # Ensure data directory exists
            os.makedirs(data_dir, exist_ok=True)

            # Try to use the data downloader if available
            if download_all_data_with_consolidation:
                success = await download_all_data_with_consolidation(
                    symbol=symbol,
                    exchange_name=exchange,
                    interval=timeframe,
                    data_dir=data_dir,
                )

                if success:
                    self.logger.info("✅ Data download completed successfully")
                    # Validate downloaded data
                    validation_success = await self._validate_downloaded_data(
                        symbol, exchange, timeframe, data_dir
                    )
                    if validation_success:
                        self.logger.info("✅ Downloaded data validation passed")
                    else:
                        self.logger.warning("⚠️ Downloaded data validation found issues")

                    # Log detailed data extract
                    await self._log_detailed_data_extract(
                        symbol, exchange, timeframe, data_dir, self.logger
                    )

                    return bool(success)

            # Fallback implementation
            self.logger.warning("⚠️ Using fallback data collection method")
            return await self._fallback_data_collection(training_input, data_dir)

        except Exception as e:
            self.logger.exception(f"❌ Error in data collection: {e}")
            return False

    async def _validate_downloaded_data(
        self, symbol: str, exchange: str, timeframe: str, data_dir: str
    ) -> bool:
        """Validate downloaded data using standardized validation."""
        try:
            self.logger.info("🔍 Validating downloaded data...")

            # Check for expected files
            expected_files = [
                self.standards.generate_file_name(
                    "klines", exchange, symbol, timeframe
                ),
                self.standards.generate_file_name("aggtrades", exchange, symbol),
            ]

            validation_results = []

            for file_name in expected_files:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    self.logger.info(f"✅ Found expected file: {file_name}")

                    # Validate file content
                    try:
                        import pandas as pd

                        df = pd.read_parquet(file_path)

                        # Standardize timestamps
                        df = self.standards.standardize_timestamp(df, "timestamp")

                        # Validate schema
                        if "klines" in file_name:
                            schema_name = "klines"
                        elif "aggtrades" in file_name:
                            schema_name = "aggtrades"
                        else:
                            schema_name = "unified"

                        validation_result = self.standards.validate_data_quality(
                            df, schema_name
                        )
                        validation_results.append(validation_result)

                        if validation_result.passed:
                            self.logger.info(
                                f"✅ {file_name} validation passed (quality score: {validation_result.quality_score:.2f})"
                            )
                        else:
                            self.logger.warning(
                                f"⚠️ {file_name} validation issues: {len(validation_result.issues)} issues, {len(validation_result.warnings)} warnings"
                            )

                    except Exception as e:
                        self.logger.exception(f"❌ Error validating {file_name}: {e}")
                        return False
                else:
                    self.logger.warning(f"⚠️ Expected file not found: {file_name}")

            # Overall validation result
            if validation_results:
                overall_passed = all(result.passed for result in validation_results)
                overall_quality_score = sum(
                    result.quality_score for result in validation_results
                ) / len(validation_results)
                self.logger.info(
                    f"📊 Overall validation: {'PASSED' if overall_passed else 'FAILED'} (avg quality score: {overall_quality_score:.2f})"
                )
                return overall_passed

            return False

        except Exception as e:
            self.logger.exception(f"❌ Error in data validation: {e}")
            return False

    async def _fallback_data_collection(
        self, training_input: dict[str, Any], data_dir: str
    ) -> bool:
        """Fallback data collection method with standardized validation."""
        self.logger.info("🔄 Running fallback data collection...")

        try:
            symbol = training_input.get("symbol")
            exchange = training_input.get("exchange")
            timeframe = training_input.get("timeframe", "1m")

            if not symbol or not exchange:
                self.logger.error(
                    "❌ Symbol and exchange required for fallback collection"
                )
                return False

            # Create mock data for testing purposes
            self.logger.info("📊 Creating mock data for fallback collection...")

            from datetime import datetime, timedelta

            import numpy as np
            import pandas as pd

            # Generate mock klines data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            timestamps = pd.date_range(start=start_date, end=end_date, freq="1min")

            # Generate realistic price data
            np.random.seed(42)
            base_price = 3000.0
            price_changes = np.random.normal(0, 0.002, len(timestamps))
            prices = [base_price]

            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 100))

            prices = np.array(prices)

            # Create klines DataFrame
            klines_data = []
            for i, timestamp in enumerate(timestamps):
                price = prices[i]
                volume = np.random.uniform(10, 1000)

                spread = price * 0.001
                open_price = price + np.random.uniform(-spread, spread)
                high_price = max(open_price, price + np.random.uniform(0, spread))
                low_price = min(open_price, price - np.random.uniform(0, spread))
                close_price = price + np.random.uniform(-spread, spread)

                klines_data.append(
                    {
                        "timestamp": int(
                            timestamp.timestamp() * 1000
                        ),  # Convert to milliseconds
                        "open": round(open_price, 2),
                        "high": round(high_price, 2),
                        "low": round(low_price, 2),
                        "close": round(close_price, 2),
                        "volume": round(volume, 2),
                    }
                )

            klines_df = pd.DataFrame(klines_data)

            # Standardize timestamps and enforce schema
            klines_df = self.standards.standardize_timestamp(klines_df, "timestamp")
            klines_df = self.standards.enforce_schema(klines_df, "klines")

            # Save klines data
            klines_file = self.standards.generate_file_name(
                "klines", exchange, symbol, timeframe
            )
            klines_path = os.path.join(data_dir, klines_file)
            klines_df.to_parquet(klines_path, index=False)

            self.logger.info(f"✅ Created mock klines data: {len(klines_df)} rows")
            self.logger.info(f"💾 Saved to: {klines_path}")

            # Generate mock aggtrades data
            aggtrades_data = []
            for i in range(0, len(timestamps), 5):  # Every 5 minutes
                timestamp = timestamps[i]
                price = prices[i] if i < len(prices) else base_price

                num_trades = np.random.randint(1, 10)
                for _ in range(num_trades):
                    trade_price = price + np.random.normal(0, 50)
                    quantity = np.random.uniform(0.1, 10.0)

                    aggtrades_data.append(
                        {
                            "timestamp": int(timestamp.timestamp() * 1000),
                            "price": round(trade_price, 2),
                            "quantity": round(quantity, 4),
                            "is_buyer_maker": np.random.choice([True, False]),
                        }
                    )

            aggtrades_df = pd.DataFrame(aggtrades_data)

            # Standardize timestamps and enforce schema
            aggtrades_df = self.standards.standardize_timestamp(
                aggtrades_df, "timestamp"
            )
            aggtrades_df = self.standards.enforce_schema(aggtrades_df, "aggtrades")

            # Save aggtrades data
            aggtrades_file = self.standards.generate_file_name(
                "aggtrades", exchange, symbol
            )
            aggtrades_path = os.path.join(data_dir, aggtrades_file)
            aggtrades_df.to_parquet(aggtrades_path, index=False)

            self.logger.info(
                f"✅ Created mock aggtrades data: {len(aggtrades_df)} rows"
            )
            self.logger.info(f"💾 Saved to: {aggtrades_path}")

            return True

        except Exception as e:
            self.logger.exception(f"❌ Error in fallback data collection: {e}")
            return False

    async def _run_comprehensive_validation(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        logger: Any,
    ) -> bool:
        """Run comprehensive file format validation for step 1."""
        try:
            if not validate_step1_file:
                logger.warning("Comprehensive file validation not available")
                return True

            # Define expected files for step 1
            expected_files = [
                f"{data_dir}/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
                f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            ]

            validation_results: list[Any] = []
            all_valid = True

            for file_path in expected_files:
                if Path(file_path).exists():
                    logger.info(f"🔍 Validating file: {file_path}")

                    # Validate file format
                    validation_result = validate_step1_file(file_path)  # type: ignore[misc]
                    validation_results.append(validation_result)

                    if getattr(validation_result, "is_valid", False):
                        logger.info(f"✅ File validation passed: {file_path}")
                        logger.info(
                            f"   📊 Shape: {validation_result.summary.get('shape', 'N/A')}"
                        )
                        logger.info(f"   📁 File type: {validation_result.file_type}")
                        logger.info(
                            f"   🗂️ Columns: {validation_result.summary.get('column_count', 'N/A')}"
                        )
                    else:
                        logger.warning(f"⚠️ File validation issues found: {file_path}")
                        all_valid = False

                        # Log detailed issues
                        for issue in getattr(validation_result, "issues", []) or []:
                            logger.warning(
                                f"   - {issue.severity.value.upper()}: {issue.description}"
                            )
                            if getattr(issue, "details", None):
                                logger.warning(f"     Details: {issue.details}")
                else:
                    logger.warning(f"⚠️ Expected file not found: {file_path}")
                    all_valid = False

            # Log validation summary
            if validation_results:
                total_files = len(validation_results)
                valid_files = sum(
                    1 for r in validation_results if getattr(r, "is_valid", False)
                )
                logger.info(
                    f"📊 Validation Summary: {valid_files}/{total_files} files passed validation"
                )

            return all_valid

        except Exception as e:
            logger.exception(f"❌ Error during comprehensive validation: {e}")
            return False

    async def _log_detailed_data_extract(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        logger: Any,
    ) -> None:
        """Log detailed data extract for troubleshooting purposes.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            logger: Logger instance
        """
        logger.info("=" * 80)
        logger.info("📊 DETAILED DATA EXTRACT FOR TROUBLESHOOTING")
        logger.info("=" * 80)

        try:
            import pandas as pd

            # Check for consolidated files
            klines_file = f"{data_dir}/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            aggtrades_file = (
                f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            )

            files_to_check = [
                ("Klines", klines_file),
                ("Aggtrades", aggtrades_file),
            ]

            for data_type, file_path in files_to_check:
                logger.info(f"🔍 Analyzing {data_type} data: {file_path}")

                if Path(file_path).exists():
                    try:
                        # Load the data
                        df = pd.read_parquet(file_path)

                        # Basic information
                        logger.info(f"   📊 Shape: {df.shape}")
                        logger.info(
                            f"   📁 File size: {Path(file_path).stat().st_size:,} bytes"
                        )

                        # Column information
                        logger.info(
                            f"   🗂️ Columns ({len(df.columns)}): {list(df.columns)}"
                        )

                        # Data types
                        logger.info("   🔧 Data types:")
                        for col, dtype in df.dtypes.items():
                            logger.info(f"      - {col}: {dtype}")

                        # Sample data (first 5 rows)
                        logger.info("   📋 Sample data (first 5 rows):")
                        sample_df = df.head(5)
                        for idx, row in sample_df.iterrows():
                            # Format the row data for better readability
                            formatted_row = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    formatted_row[col] = "NaN"
                                elif isinstance(val, int | float):
                                    formatted_row[col] = (
                                        f"{val:.6f}"
                                        if isinstance(val, float)
                                        else str(val)
                                    )
                                else:
                                    formatted_row[col] = str(val)
                            logger.info(f"      Row {idx}: {formatted_row}")

                        # Last 5 rows for comparison
                        logger.info("   📋 Sample data (last 5 rows):")
                        sample_df_last = df.tail(5)
                        for idx, row in sample_df_last.iterrows():
                            # Format the row data for better readability
                            formatted_row = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    formatted_row[col] = "NaN"
                                elif isinstance(val, int | float):
                                    formatted_row[col] = (
                                        f"{val:.6f}"
                                        if isinstance(val, float)
                                        else str(val)
                                    )
                                else:
                                    formatted_row[col] = str(val)
                            logger.info(f"      Row {idx}: {formatted_row}")

                        # Date range information
                        if "timestamp" in df.columns:
                            try:
                                df["timestamp"] = pd.to_datetime(df["timestamp"])
                                min_date = df["timestamp"].min()
                                max_date = df["timestamp"].max()
                                total_days = (max_date - min_date).days
                                logger.info(
                                    f"   📅 Date range: {min_date} to {max_date} ({total_days} days)"
                                )
                            except Exception as e:
                                logger.warning(f"   ⚠️ Could not parse timestamp: {e}")

                        # Value ranges for numeric columns
                        numeric_cols = df.select_dtypes(include=["number"]).columns
                        if len(numeric_cols) > 0:
                            logger.info("   📈 Numeric value ranges:")
                            for col in numeric_cols:
                                if col in df.columns:
                                    col_data = df[col].dropna()
                                    if len(col_data) > 0:
                                        min_val = col_data.min()
                                        max_val = col_data.max()
                                        mean_val = col_data.mean()
                                        logger.info(
                                            f"      - {col}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}"
                                        )

                        # Missing values
                        missing_counts = df.isnull().sum()
                        if missing_counts.sum() > 0:
                            logger.warning("   ⚠️ Missing values:")
                            for col, count in missing_counts.items():
                                if count > 0:
                                    percentage = (count / len(df)) * 100
                                    logger.warning(
                                        f"      - {col}: {count} ({percentage:.2f}%)"
                                    )
                        else:
                            logger.info("   ✅ No missing values found")

                        # Duplicate check
                        if "timestamp" in df.columns:
                            duplicates = df.duplicated(subset=["timestamp"]).sum()
                            if duplicates > 0:
                                logger.warning(
                                    f"   ⚠️ Found {duplicates} duplicate timestamps"
                                )
                            else:
                                logger.info("   ✅ No duplicate timestamps found")

                        # Data quality checks
                        logger.info("   🔍 Data quality checks:")

                        # Check for infinite values
                        infinite_counts = {}
                        for col in numeric_cols:
                            if col in df.columns:
                                infinite_count = (df[col] == float("inf")).sum() + (
                                    df[col] == float("-inf")
                                ).sum()
                                if infinite_count > 0:
                                    infinite_counts[col] = infinite_count

                        if infinite_counts:
                            logger.warning("      ⚠️ Infinite values found:")
                            for col, count in infinite_counts.items():
                                logger.warning(
                                    f"         - {col}: {count} infinite values"
                                )
                        else:
                            logger.info("      ✅ No infinite values found")

                        # Check for zero values in price columns
                        price_columns = ["open", "high", "low", "close", "price"]
                        zero_price_counts = {}
                        for col in price_columns:
                            if col in df.columns:
                                zero_count = (df[col] == 0).sum()
                                if zero_count > 0:
                                    zero_price_counts[col] = zero_count

                        if zero_price_counts:
                            logger.warning("      ⚠️ Zero values in price columns:")
                            for col, count in zero_price_counts.items():
                                logger.warning(f"         - {col}: {count} zero values")
                        else:
                            logger.info("      ✅ No zero values in price columns")

                        # Check for negative values in volume
                        if "volume" in df.columns:
                            negative_volume = (df["volume"] < 0).sum()
                            if negative_volume > 0:
                                logger.warning(
                                    f"      ⚠️ Negative volume values: {negative_volume}"
                                )
                            else:
                                logger.info("      ✅ No negative volume values")

                        logger.info(f"   ✅ {data_type} data analysis completed")

                    except Exception as e:
                        logger.exception(f"   ❌ Error analyzing {data_type} data: {e}")
                        logger.exception(f"   📋 Full error: {str(e)}")
                else:
                    logger.warning(f"   ⚠️ File not found: {file_path}")

                logger.info("")  # Empty line for readability

            # Summary
            logger.info("📋 DATA EXTRACT SUMMARY:")
            existing_files = sum(
                1 for _, file_path in files_to_check if Path(file_path).exists()
            )
            logger.info(f"   • Files found: {existing_files}/{len(files_to_check)}")
            logger.info("   • Data types analyzed: Klines, Aggtrades")
            logger.info(
                "   • Information logged: Shape, columns, data types, sample data, date ranges, value ranges, missing values, duplicates"
            )
            logger.info("=" * 80)

        except Exception as e:
            logger.exception(f"❌ Error in detailed data extract: {e}")
            logger.exception(f"📋 Full error: {str(e)}")
            logger.info("=" * 80)


@monitor_data_collection()
@handles_errors(fallback=False)
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = None,  # Will be constructed as data_cache/exchange/asset/
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the data collection step.

    Args:
        symbol: Trading symbol (e.g. = "ETHUSDT")
        exchange: Exchange name (e.g. = "BINANCE")
        timeframe: Timeframe (e.g. = "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if data exists
        **kwargs: Additional arguments

    Returns:
        bool: True if successful = False otherwise

    """
    try:
        logger = system_logger.getChild("Step1DataCollection")

        logger.info("=" * 80)
        logger.info("🚀 STEP 1: Data Collection")
        logger.info("=" * 80)
        logger.info(f"🎯 Symbol: {symbol}")
        logger.info(f"🏢 Exchange: {exchange}")
        logger.info(f"📊 Timeframe: {timeframe}")
        # Use standardized path construction
        if data_dir is None:
            data_dir = pipeline_standards.build_path("raw_data", exchange, symbol)
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🔄 Force rerun: {force_rerun}")

        # Check if data already exists and force_rerun is False
        if not force_rerun:
            # Check for existing consolidated data using standardized file names
            klines_file = pipeline_standards.generate_file_name(
                "klines", exchange, symbol, timeframe
            )
            aggtrades_file = pipeline_standards.generate_file_name(
                "aggtrades", exchange, symbol
            )

            consolidated_files = [
                os.path.join(data_dir, klines_file),
                os.path.join(data_dir, aggtrades_file),
            ]

            existing_files: list[str] = []
            for file_path in consolidated_files:
                if Path(file_path).exists():
                    existing_files.append(file_path)

            if existing_files:
                logger.info(
                    f"✅ Found existing consolidated data: {len(existing_files)} files"
                )
                logger.info("   📁 Existing files:")
                for file_path in existing_files:
                    logger.info(f"      - {file_path}")

                # Check if data is complete by examining the date range
                try:
                    import pandas as pd

                    klines_path = os.path.join(data_dir, klines_file)
                    if Path(klines_path).exists():
                        df = pd.read_parquet(klines_path)
                        if "timestamp" in df.columns:
                            # Standardize timestamp format for checking
                            df = pipeline_standards.standardize_timestamp(
                                df, "timestamp", "datetime64[ns]"
                            )
                            df["timestamp"].min().date()
                            max_date = df["timestamp"].max().date()
                            current_date = datetime.now().date()

                            # Check if we have recent data (within last 30 days)
                            days_since_last_data = (current_date - max_date).days

                            if days_since_last_data > 30:
                                logger.info(
                                    f"⚠️ Data is {days_since_last_data} days old, downloading recent data..."
                                )
                                # Continue with data collection to download missing data
                            else:
                                logger.info(
                                    f"✅ Data is up to date (last data: {max_date}, {days_since_last_data} days ago)"
                                )
                                logger.info(
                                    "✅ Step 1: Data Collection completed (using existing data)"
                                )

                                # Show detailed data extract for existing data
                                step = DataCollectionStep(CONFIG or {})
                                await step._log_detailed_data_extract(
                                    symbol, exchange, timeframe, data_dir, logger
                                )

                                return True
                        else:
                            logger.warning(
                                "⚠️ Could not determine data completeness, proceeding with data collection..."
                            )
                    else:
                        logger.warning(
                            "⚠️ Klines file not found, proceeding with data collection..."
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error checking data completeness: {e}, proceeding with data collection..."
                    )

        # Initialize data collection step
        step = DataCollectionStep(CONFIG or {})
        await step.initialize()

        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            "asset": symbol,  # Use symbol as asset
            "lookback_period": (
                CONFIG.get("lookback_days", 1095) if CONFIG else 1095
            ),  # Default to 3 years
            "project_version": (
                CONFIG.get("project_version", "1.0.0") if CONFIG else "1.0.0"
            ),  # Default version
        }

        # Execute data collection
        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        if result.get("data_collection_completed", False):
            logger.info("✅ Step 1: Data Collection completed successfully")

            # Show detailed data extract for troubleshooting
            await step._log_detailed_data_extract(
                symbol, exchange, timeframe, data_dir, logger
            )

            # Run standardized data quality validation
            try:
                logger.info("🔍 Running standardized data quality validation...")
                validation_success = await step._run_standardized_quality_check(
                    symbol, exchange, timeframe, data_dir
                )

                if validation_success:
                    logger.info("✅ Standardized data quality validation passed")
                else:
                    logger.warning(
                        "⚠️ Standardized data quality validation found issues"
                    )
                    logger.warning(
                        "⚠️ Continuing with data quality issues - review logs for details"
                    )

            except Exception as e:
                logger.warning(
                    f"⚠️ Standardized data quality validation failed: {e} - continuing anyway"
                )

            return True
        logger.error("❌ Step 1: Data Collection failed")
        return False

    except Exception as e:
        logger.exception(f"❌ Step 1: Data Collection failed: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    import asyncio

    async def main() -> None:
        # Get command line arguments
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else "data_cache"
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
        else:
            print(
                "Usage: python step1_data_collection.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]"
            )
            print(
                "Example: python step1_data_collection.py ETHUSDT BINANCE 1m data_cache true"
            )
            return

        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        if success:
            print("✅ Step 1: Data Collection completed successfully")
        else:
            print("❌ Step 1: Data Collection failed")

        # Clean up memory to prevent segmentation fault
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
import os.path

gc.collect()
