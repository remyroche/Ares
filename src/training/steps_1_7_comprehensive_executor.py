#!/usr/bin/env python3
"""
Comprehensive Steps 1-7 Executor with Enhanced Data Quality Management.

This script systematically executes steps 1-7 of the enhanced training pipeline, ensuring data compatibility, quality, format compatibility, and proper indexing
at every step with comprehensive validation and error handling.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.mlflow_logging import (
    log_step_report,
    create_detailed_step_report, log_step_metrics, log_step_dataframe_with_standardized_name, log_step_artifact_with_standardized_name
)

# Import all step classes
from src.training.steps.step01_data_collection import DataCollectionStep
from src.training.steps.step01_5_data_converter import DataConverterStep
from src.training.steps.step02_data_reading import DataReadingStep
from src.training.steps.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.step04_regime_data_splitting import RegimeDataSplittingStep
from src.training.steps.step05_labeling import LabelingStep
from src.training.steps.step06_feature_engineering import FeatureEngineeringStep
from src.training.steps.step07_enhanced_matrix_operations import Step7EnhancedMatrixOperations

# Import validators
from src.training.steps.step01_data_collection_validator import run_validator as validate_step1
from src.training.steps.step01_5_data_converter_validator import run_validator as validate_step1_5
from src.training.steps.step02_data_reading_validator import run_validator as validate_step2
from src.training.steps.step03_hmm_regime_discovery_validator import run_validator as validate_step3
from src.training.steps.step04_regime_data_splitting_validator import run_validator as validate_step4
from src.training.steps.step05_labeling_validator import run_validator as validate_step5
from src.training.steps.step06_feature_engineering_validator import run_validator as validate_step6
from src.training.steps.step07_enhanced_matrix_operations_validator import run_validator as validate_step7


class Steps1To7ComprehensiveExecutor:
                """
    Comprehensive executor for steps 1-7 with enhanced data quality management.

    This class ensures:
                - Data compatibility across all steps
    - Data quality validation at each step
    - Format compatibility and standardization
    - Proper indexing and temporal alignment
    - Comprehensive error handling and recovery
    - Detailed logging and monitoring
    """

        self.logger = system_logger.getChild("Steps1To7Executor")
        self.pipeline_state = {}
        self.execution_timings = {}
        self.data_quality_scores = {}
        self.errors_encountered = []

        # Initialize step instances
        self.steps = {
            "step1": DataCollectionStep(config),
            "step01_5": DataConverterStep(config),
            "step2": DataReadingStep(config),
            "step3": HMMRegimeDiscoveryStep(config),
            "step4": RegimeDataSplittingStep(config),
            "step5": LabelingStep(config),
            "step6": FeatureEngineeringStep(config),
            "step7": Step7EnhancedMatrixOperations(config)
        }

        # Initialize validators
        self.validators = {
            "step1": validate_step1,
            "step01_5": validate_step1_5,
            "step2": validate_step2,
            "step3": validate_step3,
            "step4": validate_step4,
            "step5": validate_step5,
            "step6": validate_step6,
            "step7": validate_step7
        }

        self.logger.info("🚀 Steps 1-7 Comprehensive Executor initialized")

    async def initialize_all_steps(...) -> ...:
    """..."""
                self.logger.info("🔧 Initializing all pipeline steps...")

                self.errors_encountered.append(f"{step_name}_initialization_error: {str(e)}")
                return False

        self.logger.info("✅ All steps initialized successfully")
        return True

            "compatible": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }

        try:
            if data is None:
validation_result["compatible"] = False
                validation_result["issues"].append("Data is None")
                return validation_result

                if data.empty:
validation_result["compatible"] = False
                    validation_result["issues"].append("DataFrame is empty")

                # Check required columns based on step
                required_columns = self._get_required_columns_for_step(step_name)
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    validation_result["issues"].append(f"Missing required columns: {missing_columns}")

                # Check data types
                type_issues = self._validate_data_types(data, step_name)
                if type_issues:

                # Check temporal indexing
                index_issues = self._validate_temporal_index(data, step_name)
                if index_issues:

                # Check for null values
                null_counts = data.isnull().sum()
                if null_counts.sum() > 0:
                validation_result["warnings"].append(f"Found null values: {null_counts.to_dict()}")

            validation_result["issues"].append(f"Validation error: {str(e)}")

        return validation_result

            "step1": ["timestamp", "open", "high", "low", "close", "volume"],
            "step01_5": ["timestamp", "open", "high", "low", "close", "volume"],
            "step2": ["timestamp", "open", "high", "low", "close", "volume"],
            "step3": ["timestamp", "open", "high", "low", "close", "volume"],
            "step4": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"],
            "step5": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"],
            "step6": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"],
            "step7": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"]
        }
        return column_requirements.get(step_name, [])

            "step1": ["symbol", "exchange", "timeframe", "data_dir"],
            "step01_5": ["symbol", "exchange", "timeframe", "data_dir"],
            "step2": ["symbol", "exchange", "timeframe", "data_dir"],
            "step3": ["symbol", "exchange", "timeframe", "data_dir"],
            "step4": ["symbol", "exchange", "timeframe", "data_dir"],
            "step5": ["symbol", "exchange", "timeframe", "data_dir"],
            "step6": ["symbol", "exchange", "timeframe", "data_dir"],
            "step7": ["symbol", "exchange", "timeframe", "data_dir"]
        }
        return key_requirements.get(step_name, [])


        expected_types = {
            "timestamp": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64"
        }


        return issues

    def _validate_temporal_index(...) -> ...:
    """..."""
issues = []

        if "timestamp" in data.columns:
# Check if timestamp is sorted
            if not data["timestamp"].is_monotonic_increasing:
issues.append("Timestamp column is not monotonically increasing")

            # Check for duplicate timestamps
            if data["timestamp"].duplicated().any():
                issues.append("Found duplicate timestamps")

            # Check for gaps in data
            if len(data) > 1:
                time_diff = data["timestamp"].diff().dropna()
                if time_diff.std() > time_diff.mean() * 2:
issues.append("Large variations in time intervals detected")

        return issues

            "improvements": [],
            "passed": True
        }

        try:
                quality_metrics = {}

                # Completeness
                completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
                quality_metrics["completeness"] = completeness

                # Consistency
                consistency_score = self._calculate_consistency_score(data)
                quality_metrics["consistency"] = consistency_score

                # Validity
                validity_score = self._calculate_validity_score(data)
                quality_metrics["validity"] = validity_score

                # Overall quality score
                quality_result["quality_score"] = np.mean(list(quality_metrics.values()))
                quality_result["passed"] = quality_result["quality_score"] >= 0.8

                # Identify issues
                if completeness < 0.95:
quality_result["issues"].append(f"Low completeness: {completeness:.3f}")

                if consistency_score < 0.8:
quality_result["issues"].append(f"Low consistency: {consistency_score:.3f}")

                if validity_score < 0.9:
quality_result["issues"].append(f"Low validity: {validity_score:.3f}")

                # Suggest improvements
                if data.isnull().any().any():
quality_result["improvements"].append("Consider imputation for missing values")

                if len(data) < 1000:
                quality_result["improvements"].append("Consider collecting more data")

        except Exception as e:
            quality_result["issues"].append(f"Quality assessment error: {str(e)}")

        return quality_result

            # Check price relationships
            price_consistency = 0
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                valid_prices = (
                    (data["high"] >= data["low"]) &
                    (data["high"] >= data["open"]) &
                    (data["high"] >= data["close"]) &
                    (data["low"] <= data["open"]) &
                    (data["low"] <= data["close"])
                )
                price_consistency = valid_prices.mean()

            # Check volume consistency
            volume_consistency = 1.0
            if "volume" in data.columns:
volume_consistency = (data["volume"] >= 0).mean()

            return (price_consistency + volume_consistency) / 2

        except Exception:
                return 0.5

                ).mean()
                validity_checks.append(price_validity)

            # Check for reasonable price ranges
            if "close" in data.columns:
                price_range_validity = (
                    (data["close"] > 0) & (data["close"] < 1e6)
                ).mean()
                validity_checks.append(price_range_validity)

            # Check for reasonable volumes
            if "volume" in data.columns:
                volume_validity = (
                    (data["volume"] >= 0) & (data["volume"] < 1e12)
                ).mean()
                validity_checks.append(volume_validity)


        except Exception:
                return 0.5

            "recommendations": []
        }

        try:

                # Ensure proper indexing
                index_issues = self._ensure_proper_indexing(data, step_name)
                if index_issues:

                # Ensure column naming consistency
                naming_issues = self._ensure_column_naming(data, step_name)
                if naming_issues:
            format_result["issues"].append(f"Format compatibility error: {str(e)}")

        return format_result

                conversions.append("Converted timestamp to datetime64[ns]")
            except Exception:
                

        # Convert numeric columns to float64
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
                    conversions.append(f"Converted {col} to float64")
                except Exception:
                

        return conversions


        if "timestamp" in data.columns:
# Set timestamp as index if not already
            if data.index.name != "timestamp":
                    issues.append("Set timestamp as index")
                except Exception:
                issues.append("Failed to set timestamp as index")

            # Sort by timestamp
            if not data.index.is_monotonic_increasing:
                    issues.append("Sorted data by timestamp")
                except Exception:
                issues.append("Failed to sort data by timestamp")

        return issues


        # Standardize column names to lowercase
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for expected_col in expected_columns:
                if expected_col not in data.columns:
# Check for case variations
                for col in data.columns:
                        issues.append(f"Renamed {col} to {expected_col}")
                        break

        return issues

        self.logger.info(f"🚀 Executing {step_name}...")

        step_result = {
            "success": False,
            "data": None,
            "validation_passed": False,
            "quality_score": 0.0,
            "execution_time": 0.0,
            "errors": [],
            "warnings": []
        }

        try:
            # Execute the step
            step_instance = self.steps[step_name]
            step_data = await step_instance.execute(training_input, self.pipeline_state)

            # Update pipeline state
            self.pipeline_state.update(step_data)

            # Validate step output
            validation_result = await self.validators[step_name](training_input, self.pipeline_state)
            step_result["validation_passed"] = validation_result.get("validation_passed", False)

            if step_result["validation_passed"]:
                    if not compatibility_result["compatible"]:
step_result["warnings"].extend(compatibility_result["issues"])

                # Ensure data quality
                if "data" in step_data:
                    quality_result = await self.ensure_data_quality(step_name, step_data["data"])
                    step_result["quality_score"] = quality_result["quality_score"]
                    if not quality_result["passed"]:
step_result["warnings"].extend(quality_result["issues"])

                # Ensure format compatibility
                if "data" in step_data:
                    format_result = await self.ensure_format_compatibility(step_name, step_data["data"])
                    if not format_result["compatible"]:
step_result["warnings"].extend(format_result["issues"])

                step_result["success"] = True
                step_result["data"] = step_data.get("data")

            else:
step_result["errors"].extend(validation_result.get("errors", []))

        except Exception as e:
            self.errors_encountered.append(f"{step_name}_execution_error: {str(e)}")

        finally:
step_result["execution_time"] = time.time() - step_start_time
            self.execution_timings[step_name] = step_result["execution_time"]

            if step_result["success"]:
self.data_quality_scores[step_name] = step_result["quality_score"]
                self.logger.info(f"✅ {step_name} completed successfully (Quality: {step_result['quality_score']:.3f})")
            else:
                self.logger.error(f"❌ {step_name} failed: {step_result['errors']}")

        return step_result

        self.logger.info("🚀 Starting comprehensive pipeline execution (Steps 1-7)...")

        # Initialize all steps
        if not await self.initialize_all_steps():
                return {"success": False, "error": "Failed to initialize steps"}

        # Execute steps in order
        step_order = ["step1", "step01_5", "step2", "step3", "step4", "step5", "step6", "step7"]
        step_results = {}

        for step_name in step_order:
            step_results[step_name] = step_result

            if not step_result["success"]:
                self.logger.error(f"❌ Pipeline failed at {step_name}")
                break

        # Calculate overall pipeline metrics
        pipeline_result = {
        }

        # Log comprehensive report
        await self._log_pipeline_report(training_input, pipeline_result)

        return pipeline_result

            exchange = training_input.get("exchange", "UNKNOWN")
            timeframe = training_input.get("timeframe", "1m")

            # Create detailed report
            report_data = {
                "pipeline_success": pipeline_result["success"],
                "total_execution_time": pipeline_result["total_execution_time"],
                "average_quality_score": pipeline_result["average_quality_score"],
                "step_results": pipeline_result["step_results"],
                "errors_encountered": pipeline_result["errors_encountered"],
                "execution_timings": self.execution_timings,
                "data_quality_scores": self.data_quality_scores
            }

            # Log the report
            report_name = log_step_report(
                config=self.config,
                step_name="steps_1_7_comprehensive_execution",
                report_data=report_data,
                report_type="pipeline_execution_report",
                additional_metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "pipeline_success": pipeline_result["success"],
                    "total_steps": 7,
                    "successful_steps": sum(1 for result in pipeline_result["step_results"].values() if result["success"])
                }
            )

            self.logger.info(f"✅ Logged comprehensive pipeline report: {report_name}")

        except Exception as e:


async def main(...):
"""Main execution function."""
    # Example configuration
    config = {
        "SYMBOL": "ETHUSDT",
        "EXCHANGE": "BINANCE",
        "TIMEFRAME": "1m",
        "DATA_DIR": "data_cache",
        "LOOKBACK_DAYS": 1095,
        "project_version": "1_2_3"
    }

    # Example training input
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
        "data_dir": "data_cache",
        "lookback_days": 1095
    }

    # Initialize and execute pipeline
    executor = Steps1To7ComprehensiveExecutor(config)
    result = await executor.execute_pipeline(training_input)

    # Print results
    print("\n" + "="*80)
    print("PIPELINE EXECUTION RESULTS")
    print("="*80)
    print(f"Overall Success: {'✅' if result['success'] else '❌'}")
    print(f"Total Execution Time: {result['total_execution_time']:.2f} seconds")
    print(f"Average Quality Score: {result['average_quality_score']:.3f}")
    print(f"Errors Encountered: {len(result['errors_encountered'])}")

    print("\nStep Results:")
        quality = f"Quality: {step_result['quality_score']:.3f}" if step_result['quality_score'] > 0 else "N/A"
        print(f"  {step_name}: {status} ({quality}) - {step_result['execution_time']:.2f}s")

    if result['errors_encountered']:
                print("\nErrors:")
        for error in result['errors_encountered']:
                print(f"  - {error}")


if __name__ == "__main__":
asyncio.run(main())