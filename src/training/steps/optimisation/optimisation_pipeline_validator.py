#!/usr/bin/env python3
"""
Optimisation Pipeline Validator

Comprehensive validator for the optimisation pipeline with:
- Step dependency validation
- Data integrity checks
- Output verification
- Performance monitoring
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.utils.pipeline_protection_framework import (
    DataValidator,
    ValidationLevel,
    OperationType,
    PipelineState,
    DataIntegrityCheck
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import missing, failed, validation_error


class OptimisationPipelineValidator(BaseValidator):
    """Comprehensive validator for the optimisation pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__("optimisation_pipeline", config)
        self.data_validator = DataValidator(config)
        self.validation_level = ValidationLevel(config.get("validation_level", "comprehensive"))
        
    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> bool:
        """Validate the complete optimisation pipeline.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Starting comprehensive optimisation pipeline validation...")
        
        validation_results = {}
        
        try:
            # 1. Validate input parameters
            input_validation = await self._validate_input_parameters(training_input)
            validation_results["input_validation"] = input_validation
            
            if not input_validation["passed"]:
                self.logger.error("❌ Input parameter validation failed")
                return False
            
            # 2. Validate pipeline dependencies
            dependency_validation = await self._validate_pipeline_dependencies(pipeline_state)
            validation_results["dependency_validation"] = dependency_validation
            
            if not dependency_validation["passed"]:
                self.logger.error("❌ Pipeline dependency validation failed")
                return False
            
            # 3. Validate data availability and integrity
            data_validation = await self._validate_data_availability(training_input, pipeline_state)
            validation_results["data_validation"] = data_validation
            
            if not data_validation["passed"]:
                self.logger.error("❌ Data validation failed")
                return False
            
            # 4. Validate step outputs
            output_validation = await self._validate_step_outputs(pipeline_state)
            validation_results["output_validation"] = output_validation
            
            if not output_validation["passed"]:
                self.logger.error("❌ Step output validation failed")
                return False
            
            # 5. Validate performance metrics
            performance_validation = await self._validate_performance_metrics(pipeline_state)
            validation_results["performance_validation"] = performance_validation
            
            if not performance_validation["passed"]:
                self.logger.warning("⚠️ Performance validation failed - continuing with warnings")
            
            # 6. Validate final results
            results_validation = await self._validate_final_results(pipeline_state)
            validation_results["results_validation"] = results_validation
            
            if not results_validation["passed"]:
                self.logger.error("❌ Final results validation failed")
                return False
            
            # Store validation results
            self.validation_results = validation_results
            
            self.logger.info("✅ Optimisation pipeline validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Optimisation pipeline validation failed: {e}")
            return False
    
    async def _validate_input_parameters(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters."""
        try:
            self.logger.info("🔍 Validating input parameters...")
            
            # Required parameters
            required_params = ["symbol", "exchange", "timeframe", "data_dir"]
            missing_params = [param for param in required_params if param not in training_input]
            
            if missing_params:
                return {
                    "passed": False,
                    "error": f"Missing required parameters: {missing_params}",
                    "missing_params": missing_params
                }
            
            # Validate parameter values
            symbol = training_input["symbol"]
            exchange = training_input["exchange"]
            timeframe = training_input["timeframe"]
            data_dir = training_input["data_dir"]
            
            # Symbol validation
            if not isinstance(symbol, str) or len(symbol) < 3:
                return {
                    "passed": False,
                    "error": f"Invalid symbol: {symbol}",
                    "symbol": symbol
                }
            
            # Exchange validation
            valid_exchanges = ["BINANCE", "MEXC", "GATEIO"]
            if exchange not in valid_exchanges:
                return {
                    "passed": False,
                    "error": f"Invalid exchange: {exchange}. Valid exchanges: {valid_exchanges}",
                    "exchange": exchange
                }
            
            # Timeframe validation
            valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            if timeframe not in valid_timeframes:
                return {
                    "passed": False,
                    "error": f"Invalid timeframe: {timeframe}. Valid timeframes: {valid_timeframes}",
                    "timeframe": timeframe
                }
            
            # Data directory validation
            if not os.path.exists(data_dir):
                return {
                    "passed": False,
                    "error": f"Data directory does not exist: {data_dir}",
                    "data_dir": data_dir
                }
            
            return {
                "passed": True,
                "message": "Input parameters validation passed",
                "validated_params": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "data_dir": data_dir
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Input parameter validation error: {str(e)}"
            }
    
    async def _validate_pipeline_dependencies(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline dependencies."""
        try:
            self.logger.info("🔍 Validating pipeline dependencies...")
            
            # Required previous steps for optimisation
            required_steps = [
                "step1_data_collection",
                "step2_data_reading", 
                "step3_hmm_regime_discovery",
                "step4_regime_data_splitting",
                "step5_labeling",
                "step6_feature_engineering",
                "step9_hmm_based_training"
            ]
            
            missing_steps = []
            failed_steps = []
            
            for step in required_steps:
                step_result = pipeline_state.get(step, {})
                
                if not step_result:
                    missing_steps.append(step)
                elif not step_result.get("success", False):
                    failed_steps.append(step)
            
            if missing_steps:
                return {
                    "passed": False,
                    "error": f"Missing required steps: {missing_steps}",
                    "missing_steps": missing_steps
                }
            
            if failed_steps:
                return {
                    "passed": False,
                    "error": f"Failed required steps: {failed_steps}",
                    "failed_steps": failed_steps
                }
            
            return {
                "passed": True,
                "message": "Pipeline dependencies validation passed",
                "validated_steps": required_steps
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Pipeline dependency validation error: {str(e)}"
            }
    
    async def _validate_data_availability(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data availability and integrity."""
        try:
            self.logger.info("🔍 Validating data availability...")
            
            symbol = training_input["symbol"]
            exchange = training_input["exchange"]
            data_dir = training_input["data_dir"]
            
            # Expected data files
            expected_files = [
                f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet",
                f"{data_dir}/{exchange}_{symbol}_regime_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_feature_engineered_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_trained_models.pkl"
            ]
            
            missing_files = []
            corrupted_files = []
            
            for file_path in expected_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                else:
                    # Check file integrity
                    try:
                        if file_path.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                            if df.empty:
                                corrupted_files.append(f"{file_path} (empty)")
                        elif file_path.endswith('.pkl'):
                            with open(file_path, 'rb') as f:
                                import pickle
                                data = pickle.load(f)
                                if not data:
                                    corrupted_files.append(f"{file_path} (empty)")
                    except Exception as e:
                        corrupted_files.append(f"{file_path} (corrupted: {str(e)})")
            
            if missing_files:
                return {
                    "passed": False,
                    "error": f"Missing required data files: {missing_files}",
                    "missing_files": missing_files
                }
            
            if corrupted_files:
                return {
                    "passed": False,
                    "error": f"Corrupted data files: {corrupted_files}",
                    "corrupted_files": corrupted_files
                }
            
            # Validate data quality
            consolidated_file = expected_files[0]
            df = pd.read_parquet(consolidated_file)
            
            data_integrity = self.data_validator.validate_dataframe(
                df,
                min_rows=1000,  # Minimum data points for optimisation
                max_null_ratio=0.1  # Maximum 10% null values
            )
            
            if not data_integrity.passed:
                return {
                    "passed": False,
                    "error": f"Data integrity check failed: {data_integrity}",
                    "data_integrity": data_integrity
                }
            
            return {
                "passed": True,
                "message": "Data availability validation passed",
                "data_integrity": data_integrity,
                "validated_files": expected_files
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Data availability validation error: {str(e)}"
            }
    
    async def _validate_step_outputs(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step outputs."""
        try:
            self.logger.info("🔍 Validating step outputs...")
            
            # Check confidence calibration output
            confidence_result = pipeline_state.get("confidence_calibration", {})
            if not confidence_result.get("success", False):
                return {
                    "passed": False,
                    "error": "Confidence calibration step failed",
                    "confidence_result": confidence_result
                }
            
            # Check parameter optimization output
            param_optimization_result = pipeline_state.get("parameter_optimization", {})
            if not param_optimization_result.get("success", False):
                return {
                    "passed": False,
                    "error": "Parameter optimization step failed",
                    "param_optimization_result": param_optimization_result
                }
            
            # Validate output files exist
            output_files = [
                "calibrated_models.pkl",
                "optimization_results.json",
                "performance_metrics.json"
            ]
            
            missing_outputs = []
            for output_file in output_files:
                if not os.path.exists(output_file):
                    missing_outputs.append(output_file)
            
            if missing_outputs:
                return {
                    "passed": False,
                    "error": f"Missing output files: {missing_outputs}",
                    "missing_outputs": missing_outputs
                }
            
            return {
                "passed": True,
                "message": "Step outputs validation passed",
                "validated_outputs": output_files
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Step outputs validation error: {str(e)}"
            }
    
    async def _validate_performance_metrics(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance metrics."""
        try:
            self.logger.info("🔍 Validating performance metrics...")
            
            performance_metrics = pipeline_state.get("performance_metrics", {})
            
            # Check for required metrics
            required_metrics = [
                "execution_time",
                "memory_usage",
                "cpu_usage",
                "model_accuracy",
                "optimization_score"
            ]
            
            missing_metrics = [metric for metric in required_metrics if metric not in performance_metrics]
            
            if missing_metrics:
                return {
                    "passed": False,
                    "error": f"Missing performance metrics: {missing_metrics}",
                    "missing_metrics": missing_metrics
                }
            
            # Validate metric values
            execution_time = performance_metrics.get("execution_time", 0)
            if execution_time > 3600:  # More than 1 hour
                return {
                    "passed": False,
                    "error": f"Execution time too high: {execution_time}s",
                    "execution_time": execution_time
                }
            
            model_accuracy = performance_metrics.get("model_accuracy", 0)
            if model_accuracy < 0.5:  # Less than 50% accuracy
                return {
                    "passed": False,
                    "error": f"Model accuracy too low: {model_accuracy}",
                    "model_accuracy": model_accuracy
                }
            
            return {
                "passed": True,
                "message": "Performance metrics validation passed",
                "validated_metrics": performance_metrics
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Performance metrics validation error: {str(e)}"
            }
    
    async def _validate_final_results(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final results."""
        try:
            self.logger.info("🔍 Validating final results...")
            
            # Check overall pipeline success
            overall_success = pipeline_state.get("overall_success", False)
            if not overall_success:
                return {
                    "passed": False,
                    "error": "Overall pipeline success is False",
                    "overall_success": overall_success
                }
            
            # Check for critical errors
            error_log = pipeline_state.get("error_log", [])
            critical_errors = [error for error in error_log if error.get("severity") == "CRITICAL"]
            
            if critical_errors:
                return {
                    "passed": False,
                    "error": f"Critical errors found: {len(critical_errors)}",
                    "critical_errors": critical_errors
                }
            
            # Validate final output files
            final_outputs = [
                "optimized_models.pkl",
                "final_parameters.json",
                "optimization_report.json"
            ]
            
            missing_final_outputs = []
            for output_file in final_outputs:
                if not os.path.exists(output_file):
                    missing_final_outputs.append(output_file)
            
            if missing_final_outputs:
                return {
                    "passed": False,
                    "error": f"Missing final output files: {missing_final_outputs}",
                    "missing_final_outputs": missing_final_outputs
                }
            
            return {
                "passed": True,
                "message": "Final results validation passed",
                "validated_final_outputs": final_outputs
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Final results validation error: {str(e)}"
            }


class ConfidenceCalibrationValidator(BaseValidator):
    """Validator for confidence calibration step."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__("confidence_calibration", config)
        self.data_validator = DataValidator(config)

    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> bool:
        """Validate confidence calibration step."""
        self.logger.info("🔍 Validating confidence calibration...")
        
        try:
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data_cache")
            
            # Validate calibration files
            calibration_files = [
                f"{data_dir}/{exchange}_{symbol}_calibrated_models.pkl",
                f"{data_dir}/{exchange}_{symbol}_calibration_metadata.json",
                f"{data_dir}/{exchange}_{symbol}_calibration_results.json"
            ]
            
            missing_files = [f for f in calibration_files if not os.path.exists(f)]
            if missing_files:
                self.logger.error(f"❌ Missing calibration files: {missing_files}")
                return False
            
            # Validate calibration results
            results_file = calibration_files[2]
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Check calibration quality
            if not results.get("calibration_successful", False):
                self.logger.error("❌ Calibration was not successful")
                return False
            
            # Check calibration metrics
            calibration_metrics = results.get("metrics", {})
            required_metrics = ["accuracy", "precision", "recall", "f1_score"]
            
            missing_metrics = [m for m in required_metrics if m not in calibration_metrics]
            if missing_metrics:
                self.logger.error(f"❌ Missing calibration metrics: {missing_metrics}")
                return False
            
            # Validate metric values
            accuracy = calibration_metrics.get("accuracy", 0)
            if accuracy < 0.7:  # Less than 70% accuracy
                self.logger.error(f"❌ Calibration accuracy too low: {accuracy}")
                return False
            
            self.logger.info("✅ Confidence calibration validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Confidence calibration validation failed: {e}")
            return False


class ParameterOptimizationValidator(BaseValidator):
    """Validator for parameter optimization step."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__("parameter_optimization", config)
        self.data_validator = DataValidator(config)

    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> bool:
        """Validate parameter optimization step."""
        self.logger.info("🔍 Validating parameter optimization...")
        
        try:
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data_cache")
            
            # Validate optimization files
            optimization_files = [
                f"{data_dir}/{exchange}_{symbol}_optimized_parameters.json",
                f"{data_dir}/{exchange}_{symbol}_optimization_results.json",
                f"{data_dir}/{exchange}_{symbol}_optimization_metrics.json"
            ]
            
            missing_files = [f for f in optimization_files if not os.path.exists(f)]
            if missing_files:
                self.logger.error(f"❌ Missing optimization files: {missing_files}")
                return False
            
            # Validate optimization results
            results_file = optimization_files[1]
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Check optimization success
            if not results.get("optimization_successful", False):
                self.logger.error("❌ Parameter optimization was not successful")
                return False
            
            # Check optimization metrics
            optimization_metrics = results.get("metrics", {})
            required_metrics = ["best_score", "improvement", "convergence"]
            
            missing_metrics = [m for m in required_metrics if m not in optimization_metrics]
            if missing_metrics:
                self.logger.error(f"❌ Missing optimization metrics: {missing_metrics}")
                return False
            
            # Validate improvement
            improvement = optimization_metrics.get("improvement", 0)
            if improvement < 0.01:  # Less than 1% improvement
                self.logger.warning(f"⚠️ Optimization improvement is minimal: {improvement}")
            
            self.logger.info("✅ Parameter optimization validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Parameter optimization validation failed: {e}")
            return False