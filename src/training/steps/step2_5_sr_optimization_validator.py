#!/usr/bin/env python3
"""Step 2.5: S/R Detection Optimization Validator.

This module validates the S/R detection optimization step to ensure:
1. Optimization results are properly saved
2. Optimized parameters are correctly formatted
3. Configuration is updated with optimized parameters
4. All required artifacts are present
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    handle_errors,
    monitor_step_execution,
    quality_gate,
    secure_step_execution,
    validate_pipeline_step,
)
from src.utils.common_operations import safe_json_load
from src.utils.logger import system_logger

logger = system_logger.getChild("Step2_5SROptimizationValidator")


class SROptimizationValidator:
    """Validator for S/R detection optimization step."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("SROptimizationValidator")
        self.validation_results = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="sr_optimization_validation",
    )
    async def validate_step(self, symbol: str, exchange: str, timeframe: str) -> bool:
        """Validate the S/R optimization step."""
        try:
            self.logger.info("🔍 Starting S/R optimization validation...")

            validation_passed = True
            validation_details = []

            # Validate optimization results file
            results_validation = await self._validate_optimization_results()
            if not results_validation["valid"]:
                validation_passed = False
                validation_details.extend(results_validation["errors"])

            # Validate optimized parameters
            params_validation = await self._validate_optimized_parameters()
            if not params_validation["valid"]:
                validation_passed = False
                validation_details.extend(params_validation["errors"])

            # Validate configuration updates
            config_validation = await self._validate_configuration_updates()
            if not config_validation["valid"]:
                validation_passed = False
                validation_details.extend(config_validation["errors"])

            # Validate artifact quality
            quality_validation = await self._validate_artifact_quality()
            if not quality_validation["valid"]:
                validation_passed = False
                validation_details.extend(quality_validation["errors"])

            self.validation_results = {
                "valid": validation_passed,
                "details": validation_details,
                "timestamp": time.time(),
                "step": "step2_5_sr_optimization",
            }

            if validation_passed:
                self.logger.info("✅ S/R optimization validation passed")
            else:
                self.logger.error(
                    f"❌ S/R optimization validation failed: {validation_details}"
                )

            return validation_passed

        except Exception as e:
            self.logger.error(f"Failed to validate S/R optimization: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={"valid": False, "errors": ["Validation failed"]},
        context="optimization_results_validation",
    )
    async def _validate_optimization_results(self) -> Dict[str, Any]:
        """Validate optimization results file."""
        try:
            self.logger.info("📊 Validating optimization results file...")

            errors = []

            # Check if optimization results file exists
            results_file = Path("data/optimization/sr_optimization_results.json")
            if not results_file.exists():
                errors.append("Optimization results file not found")
                return {"valid": False, "errors": errors}

            # Check if SR predictor results file exists
            sr_results_file = Path("optimization_results.json")
            if not sr_results_file.exists():
                errors.append("SR predictor optimization results file not found")
                return {"valid": False, "errors": errors}

            # Validate JSON format
            try:
                results_data = safe_json_load(results_file)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON format in optimization results: {e}")
                return {"valid": False, "errors": errors}

            # Validate required fields
            required_fields = [
                "method_weights",
                "strength_weights",
                "dbscan_params",
                "timeframe_weights",
                "advanced_params",
                "performance_metrics",
                "validation_metrics",
                "metadata",
            ]

            for field in required_fields:
                if field not in results_data:
                    errors.append(f"Missing required field: {field}")

            # Validate metadata
            if "metadata" in results_data:
                metadata = results_data["metadata"]
                if (
                    "step" not in metadata
                    or metadata["step"] != "step2_5_sr_optimization"
                ):
                    errors.append("Invalid step metadata")
                if "timestamp" not in metadata:
                    errors.append("Missing timestamp in metadata")

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {e}"]}

    @handle_errors(
        exceptions=(Exception,),
        default_return={"valid": False, "errors": ["Validation failed"]},
        context="optimized_parameters_validation",
    )
    async def _validate_optimized_parameters(self) -> Dict[str, Any]:
        """Validate optimized parameters structure and values."""
        try:
            self.logger.info("⚙️ Validating optimized parameters...")

            errors = []

            # Load optimization results
            results_file = Path("data/optimization/sr_optimization_results.json")
            if not results_file.exists():
                return {
                    "valid": False,
                    "errors": ["Optimization results file not found"],
                }

            results_data = safe_json_load(results_file)

            # Validate method weights
            method_weights = results_data.get("method_weights", {})
            if not isinstance(method_weights, dict):
                errors.append("Method weights must be a dictionary")
            else:
                for method, weight in method_weights.items():
                    if not isinstance(weight, (int, float)) or weight < 0:
                        errors.append(f"Invalid method weight for {method}: {weight}")

            # Validate strength weights
            strength_weights = results_data.get("strength_weights", {})
            if not isinstance(strength_weights, dict):
                errors.append("Strength weights must be a dictionary")
            else:
                for strength, weight in strength_weights.items():
                    if not isinstance(weight, (int, float)) or weight < 0:
                        errors.append(
                            f"Invalid strength weight for {strength}: {weight}"
                        )

            # Validate DBSCAN parameters
            dbscan_params = results_data.get("dbscan_params", {})
            if not isinstance(dbscan_params, dict):
                errors.append("DBSCAN parameters must be a dictionary")
            else:
                if "eps" in dbscan_params and not isinstance(
                    dbscan_params["eps"], (int, float)
                ):
                    errors.append("DBSCAN eps must be a number")
                if "min_samples" in dbscan_params and not isinstance(
                    dbscan_params["min_samples"], int
                ):
                    errors.append("DBSCAN min_samples must be an integer")

            # Validate performance metrics
            performance_metrics = results_data.get("performance_metrics", {})
            if not isinstance(performance_metrics, dict):
                errors.append("Performance metrics must be a dictionary")
            else:
                required_metrics = ["optimization_score", "sharpe_ratio", "win_rate"]
                for metric in required_metrics:
                    if metric not in performance_metrics:
                        errors.append(f"Missing performance metric: {metric}")
                    elif not isinstance(performance_metrics[metric], (int, float)):
                        errors.append(
                            f"Invalid performance metric {metric}: {performance_metrics[metric]}"
                        )

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Parameter validation error: {e}"]}

    @handle_errors(
        exceptions=(Exception,),
        default_return={"valid": False, "errors": ["Validation failed"]},
        context="configuration_validation",
    )
    async def _validate_configuration_updates(self) -> Dict[str, Any]:
        """Validate that configuration has been updated with optimized parameters."""
        try:
            self.logger.info("🔧 Validating configuration updates...")

            errors = []

            # Check if SR configuration exists
            sr_config = self.config.get("sr_breakout_predictor", {})
            if not sr_config:
                errors.append("SR breakout predictor configuration not found")
                return {"valid": False, "errors": errors}

            # Check if use_optimized_params is enabled
            if not sr_config.get("use_optimized_params", False):
                errors.append("use_optimized_params not enabled in SR configuration")

            # Check if optimization results file path is set
            if "optimization_results_file" not in sr_config:
                errors.append(
                    "optimization_results_file path not set in SR configuration"
                )

            # Check if SR detection optimization config exists
            sr_opt_config = self.config.get("sr_detection_optimization", {})
            if not sr_opt_config:
                errors.append("SR detection optimization configuration not found")

            # Check if optimized parameters are in config
            optimized_params = [
                "optimized_method_weights",
                "optimized_strength_weights",
                "optimized_dbscan_params",
                "optimized_timeframe_weights",
                "optimized_advanced_params",
            ]

            for param in optimized_params:
                if param not in sr_opt_config:
                    errors.append(
                        f"Optimized parameter {param} not found in configuration"
                    )

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Configuration validation error: {e}"]}

    @handle_errors(
        exceptions=(Exception,),
        default_return={"valid": False, "errors": ["Validation failed"]},
        context="artifact_quality_validation",
    )
    async def _validate_artifact_quality(self) -> Dict[str, Any]:
        """Validate the quality of optimization artifacts."""
        try:
            self.logger.info("🎯 Validating artifact quality...")

            errors = []

            # Load optimization results
            results_file = Path("data/optimization/sr_optimization_results.json")
            if not results_file.exists():
                return {
                    "valid": False,
                    "errors": ["Optimization results file not found"],
                }

            results_data = safe_json_load(results_file)

            # Check performance metrics quality
            performance_metrics = results_data.get("performance_metrics", {})

            # Check optimization score
            optimization_score = performance_metrics.get("optimization_score", 0)
            if optimization_score <= 0:
                errors.append(f"Low optimization score: {optimization_score}")

            # Check Sharpe ratio
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
            if sharpe_ratio < 0.3:  # Minimum acceptable Sharpe ratio
                errors.append(f"Low Sharpe ratio: {sharpe_ratio}")

            # Check win rate
            win_rate = performance_metrics.get("win_rate", 0)
            if win_rate < 0.5:  # Minimum acceptable win rate
                errors.append(f"Low win rate: {win_rate}")

            # Check validation metrics
            validation_metrics = results_data.get("validation_metrics", {})
            cross_validation_score = validation_metrics.get("cross_validation_score", 0)
            if cross_validation_score < 0.6:  # Minimum acceptable CV score
                errors.append(f"Low cross-validation score: {cross_validation_score}")

            # Check if optimization took reasonable time
            metadata = results_data.get("metadata", {})
            optimization_time = metadata.get("optimization_time", 0)
            if optimization_time > 3600:  # More than 1 hour
                errors.append(f"Optimization took too long: {optimization_time}s")

            # Check number of trials
            n_trials = metadata.get("n_trials", 0)
            if n_trials < 10:  # Minimum number of trials
                errors.append(f"Too few optimization trials: {n_trials}")

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Quality validation error: {e}"]}

    def get_validation_results(self) -> Dict[str, Any]:
        """Get validation results."""
        return self.validation_results


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step2_5_sr_optimization_validation",
)
async def run_validation(
    config: dict[str, Any], symbol: str, exchange: str, timeframe: str
) -> bool:
    """Run validation for the S/R optimization step."""
    try:
        logger.info("🚀 Starting Step 2.5: S/R Detection Optimization Validation")

        # Create validator
        validator = SROptimizationValidator(config)

        # Run validation
        success = await validator.validate_step(symbol, exchange, timeframe)

        # Log results
        results = validator.get_validation_results()
        if success:
            logger.info(
                "✅ Step 2.5: S/R Detection Optimization Validation completed successfully"
            )
        else:
            logger.error(
                f"❌ Step 2.5: S/R Detection Optimization Validation failed: {results.get('details', [])}"
            )

        return success

    except Exception as e:
        logger.error(f"Failed to run S/R optimization validation: {e}")
        return False


if __name__ == "__main__":
    # Test the validator
    import asyncio

    # Test configuration
    test_config = {
        "sr_breakout_predictor": {
            "use_optimized_params": True,
            "optimization_results_file": "optimization_results.json",
        },
        "sr_detection_optimization": {
            "optimized_method_weights": {"fractal": 0.8, "volume": 0.6},
            "optimized_strength_weights": {"volume": 0.7, "price": 0.3},
            "optimized_dbscan_params": {"eps": 0.1, "min_samples": 5},
            "optimized_timeframe_weights": {"1m": 0.4, "5m": 0.6},
            "optimized_advanced_params": {"fibonacci_sensitivity": 0.8},
        },
    }

    # Run validation
    success = asyncio.run(run_validation(test_config, "ETHUSDT", "BINANCE", "1m"))
    print(f"Validation {'successful' if success else 'failed'}")
