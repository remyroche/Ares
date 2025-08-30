#!/usr/bin/env python3
"""Validator for Step 3: Parameter Optimization.

This module validates the parameter optimization step outputs with comprehensive
quality checks for optimization results and configuration files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import system_logger
from src.utils.validation_decorators import (
    validate_file_operation,
    validate_dataframe_operation,
    validate_step2_operation,
)

logger = system_logger.getChild("Step3ParameterOptimizationValidator")


class Step3ParameterOptimizationValidator:
    """Validator for Step 3: Parameter Optimization."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

    @validate_step2_operation
    def validate_step3_parameter_optimization(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 3: Parameter Optimization.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 3: Parameter Optimization validation")

        try:
            # Check if optimization directory exists
            optimization_dir = Path(data_dir) / "optimization"
            if not optimization_dir.exists():
                self.logger.warning(
                    f"⚠️ Optimization directory not found: {optimization_dir}"
                )
                return False

            # Validate optimization results file
            results_file = optimization_dir / "parameter_optimization_results.json"
            if not results_file.exists():
                self.logger.warning(f"⚠️ Parameter optimization results not found: {results_file}")
                return False

            # Validate results file
            if not self._validate_optimization_results(results_file):
                return False

            # Check for optimization configuration file
            config_file = optimization_dir / "parameter_optimization_config.json"
            if not config_file.exists():
                self.logger.warning(f"⚠️ Optimization configuration not found: {config_file}")
                return False

            # Validate configuration file
            if not self._validate_optimization_config(config_file):
                return False

            # Check for optimization logs
            logs_file = optimization_dir / "parameter_optimization_logs.json"
            if not logs_file.exists():
                self.logger.warning(f"⚠️ Optimization logs not found: {logs_file}")
                return False

            # Validate logs file
            if not self._validate_optimization_logs(logs_file):
                return False

            # Check for optimization metrics
            metrics_file = optimization_dir / "parameter_optimization_metrics.json"
            if metrics_file.exists():
                if not self._validate_optimization_metrics(metrics_file):
                    return False

            self.logger.info("✅ Step 3: Parameter Optimization validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Step 3 validation failed: {e}")
            return False

    @validate_file_operation
    def _validate_optimization_results(self, results_file: Path) -> bool:
        """Validate the parameter optimization results file."""
        try:
            self.logger.info(f"📊 Validating optimization results: {results_file.name}")

            with open(results_file, 'r') as f:
                results_data = json.load(f)

            # Check required fields
            required_fields = ["best_parameters", "optimization_history", "final_score"]
            missing_fields = [field for field in required_fields if field not in results_data]
            if missing_fields:
                self.logger.warning(
                    f"⚠️ Missing required fields in optimization results: {missing_fields}"
                )
                return False

            # Validate best parameters
            best_params = results_data.get("best_parameters", {})
            if not best_params:
                self.logger.warning("⚠️ Empty best parameters in optimization results")
                return False

            # Check for expected parameter types
            expected_params = ["n_components", "n_clusters", "momentum_window", "volatility_window"]
            missing_params = [param for param in expected_params if param not in best_params]
            if missing_params:
                self.logger.warning(
                    f"⚠️ Missing expected parameters in best parameters: {missing_params}"
                )
                return False

            # Validate parameter values
            for param, value in best_params.items():
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"⚠️ Invalid parameter value type for {param}: {type(value)}")
                    return False
                
                # Check reasonable ranges
                if param == "n_components" and (value < 2 or value > 20):
                    self.logger.warning(f"⚠️ Unusual n_components value: {value}")
                    return False
                elif param == "n_clusters" and (value < 5 or value > 100):
                    self.logger.warning(f"⚠️ Unusual n_clusters value: {value}")
                    return False
                elif "window" in param and (value < 5 or value > 100):
                    self.logger.warning(f"⚠️ Unusual window value for {param}: {value}")
                    return False

            # Validate optimization history
            optimization_history = results_data.get("optimization_history", [])
            if not optimization_history:
                self.logger.warning("⚠️ Empty optimization history")
                return False

            # Validate final score
            final_score = results_data.get("final_score", 0)
            if not isinstance(final_score, (int, float)):
                self.logger.warning(f"⚠️ Invalid final score type: {type(final_score)}")
                return False

            self.logger.info(f"✅ Optimization results validated: {results_file.name}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to validate optimization results {results_file.name}: {e}")
            return False

    @validate_file_operation
    def _validate_optimization_config(self, config_file: Path) -> bool:
        """Validate the parameter optimization configuration file."""
        try:
            self.logger.info(f"📊 Validating optimization config: {config_file.name}")

            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Check required fields
            required_fields = ["parameter_ranges", "optimization_method", "max_iterations"]
            missing_fields = [field for field in required_fields if field not in config_data]
            if missing_fields:
                self.logger.warning(
                    f"⚠️ Missing required fields in optimization config: {missing_fields}"
                )
                return False

            # Validate parameter ranges
            param_ranges = config_data.get("parameter_ranges", {})
            if not param_ranges:
                self.logger.warning("⚠️ Empty parameter ranges in optimization config")
                return False

            # Validate each parameter range
            for param, range_data in param_ranges.items():
                if not isinstance(range_data, dict):
                    self.logger.warning(f"⚠️ Invalid parameter range format for {param}")
                    return False
                
                if "min" not in range_data or "max" not in range_data:
                    self.logger.warning(f"⚠️ Missing min/max for parameter {param}")
                    return False
                
                if range_data["min"] >= range_data["max"]:
                    self.logger.warning(f"⚠️ Invalid range for parameter {param}: min >= max")
                    return False

            # Validate optimization method
            optimization_method = config_data.get("optimization_method", "")
            valid_methods = ["bayesian", "grid_search", "random_search", "genetic"]
            if optimization_method not in valid_methods:
                self.logger.warning(f"⚠️ Unknown optimization method: {optimization_method}")
                return False

            # Validate max iterations
            max_iterations = config_data.get("max_iterations", 0)
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                self.logger.warning(f"⚠️ Invalid max_iterations: {max_iterations}")
                return False

            self.logger.info(f"✅ Optimization config validated: {config_file.name}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to validate optimization config {config_file.name}: {e}")
            return False

    @validate_file_operation
    def _validate_optimization_logs(self, logs_file: Path) -> bool:
        """Validate the parameter optimization logs file."""
        try:
            self.logger.info(f"📊 Validating optimization logs: {logs_file.name}")

            with open(logs_file, 'r') as f:
                logs_data = json.load(f)

            # Check if it's a list
            if not isinstance(logs_data, list):
                self.logger.warning("⚠️ Optimization logs should be a list")
                return False

            # Check for log entries
            if not logs_data:
                self.logger.warning("⚠️ Empty optimization logs")
                return False

            # Validate each log entry
            for i, log_entry in enumerate(logs_data):
                if not isinstance(log_entry, dict):
                    self.logger.warning(f"⚠️ Invalid log entry format at index {i}")
                    return False
                
                # Check for basic log fields
                if "timestamp" not in log_entry or "message" not in log_entry:
                    self.logger.warning(f"⚠️ Missing timestamp or message in log entry {i}")
                    return False

            self.logger.info(f"✅ Optimization logs validated: {logs_file.name}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to validate optimization logs {logs_file.name}: {e}")
            return False

    @validate_file_operation
    def _validate_optimization_metrics(self, metrics_file: Path) -> bool:
        """Validate the parameter optimization metrics file."""
        try:
            self.logger.info(f"📊 Validating optimization metrics: {metrics_file.name}")

            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

            # Check if it's a dictionary
            if not isinstance(metrics_data, dict):
                self.logger.warning("⚠️ Optimization metrics should be a dictionary")
                return False

            # Check for metrics
            if not metrics_data:
                self.logger.warning("⚠️ Empty optimization metrics")
                return False

            # Validate common metrics
            common_metrics = ["convergence_time", "total_iterations", "best_score", "score_history"]
            for metric in common_metrics:
                if metric in metrics_data:
                    value = metrics_data[metric]
                    if metric == "convergence_time" and (not isinstance(value, (int, float)) or value < 0):
                        self.logger.warning(f"⚠️ Invalid convergence_time: {value}")
                        return False
                    elif metric == "total_iterations" and (not isinstance(value, int) or value <= 0):
                        self.logger.warning(f"⚠️ Invalid total_iterations: {value}")
                        return False
                    elif metric == "best_score" and not isinstance(value, (int, float)):
                        self.logger.warning(f"⚠️ Invalid best_score: {value}")
                        return False
                    elif metric == "score_history" and not isinstance(value, list):
                        self.logger.warning(f"⚠️ Invalid score_history: {type(value)}")
                        return False

            self.logger.info(f"✅ Optimization metrics validated: {metrics_file.name}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to validate optimization metrics {metrics_file.name}: {e}")
            return False

    def validate_step_prerequisites(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate prerequisites for Step 3."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Check if step2_data_reading output exists
            step2_output_dir = Path("data/unified")
            step2_files = list(step2_output_dir.glob(f"{exchange}/{symbol}/{timeframe}/*.parquet"))
            
            if not step2_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].append(
                    f"Step 2 data reading output not found for {exchange}_{symbol}_{timeframe}"
                )
            else:
                validation_result["details"]["step2_files_found"] = len(step2_files)
                validation_result["details"]["step2_files"] = [str(f) for f in step2_files]

            # Check if optimization configuration exists
            optimization_config = Path("data/optimization/parameter_optimization_config.json")
            if not optimization_config.exists():
                validation_result["warnings"].append(
                    f"Optimization configuration not found: {optimization_config}"
                )

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Prerequisites validation failed: {str(e)}")

        return validation_result

    def validate_step_output(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate Step 3 output files and content."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Define expected output files
            output_dir = Path("data/optimization")
            expected_files = [
                "parameter_optimization_results.json",
                "parameter_optimization_config.json",
                "parameter_optimization_logs.json"
            ]

            # Check if all expected files exist
            missing_files = []
            existing_files = []
            
            for filename in expected_files:
                file_path = output_dir / filename
                if file_path.exists():
                    existing_files.append(str(file_path))
                else:
                    missing_files.append(filename)

            if missing_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].extend([
                    f"Missing parameter optimization file: {f}" for f in missing_files
                ])
            else:
                validation_result["details"]["files_found"] = len(existing_files)
                validation_result["details"]["files"] = existing_files

            # Validate file contents
            if existing_files:
                for file_path in existing_files:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        validation_result["details"][f"{Path(file_path).stem}_keys"] = list(data.keys())
                    except Exception as e:
                        validation_result["warnings"].append(f"Could not read JSON file {file_path}: {e}")

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Output validation failed: {str(e)}")

        return validation_result


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 3: Parameter Optimization.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 3: Parameter Optimization")
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        # Initialize validator
        config = training_input.get("config", {})
        validator = Step3ParameterOptimizationValidator(config)
        
        # Validate prerequisites
        prereq_result = validator.validate_step_prerequisites(symbol, exchange, timeframe)
        
        # Validate step execution
        step_result = validator.validate_step3_parameter_optimization(
            symbol, exchange, data_dir, training_input
        )
        
        # Validate outputs
        output_result = validator.validate_step_output(symbol, exchange, timeframe)
        
        # Combine results
        validation_passed = (
            prereq_result["validation_passed"] and 
            step_result and 
            output_result["validation_passed"]
        )
        
        return {
            "step_name": "step3_parameter_optimization",
            "validation_passed": validation_passed,
            "prerequisites": prereq_result,
            "step_execution": step_result,
            "outputs": output_result,
            "warnings": prereq_result["warnings"] + output_result["warnings"],
            "errors": prereq_result["errors"] + output_result["errors"]
        }
        
    except Exception as e:
        logger.exception(f"❌ Step 3 validation failed: {e}")
        return {
            "step_name": "step3_parameter_optimization",
            "validation_passed": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the validator
    import asyncio
    
    test_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE", 
        "timeframe": "1m",
        "data_dir": "data_cache",
        "config": {}
    }
    
    test_state = {}
    
    result = asyncio.run(run_validator(test_input, test_state))
    print(json.dumps(result, indent=2))