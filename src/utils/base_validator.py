"""
Base validator class for training step validators with comprehensive error handling and emoji logging.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.warning_symbols import failed, missing, validation_error
import asyncio

# Import enhanced logging functions
try:
    from .logger import log_error_with_context, log_validation_result, log_step_progress
    from .warning_symbols import error, warning, info, success
except ImportError:
    # Fallback if imports fail
    def log_error_with_context(logger, error, context=None, operation="", recovery_attempted=False):
        logger.error(f"Error in {operation}: {error}")
    
    def log_validation_result(logger, validator_name, result, details="", metrics=None):
        status = "PASSED" if result else "FAILED"
        logger.info(f"Validation {status} | {validator_name}")
    
    def log_step_progress(logger, step_name, step_number, total_steps, status="running", details="", context=None):
        logger.info(f"Step {step_number}/{total_steps} | {step_name} | {status}")
    
    def error(msg): return f"❌ {msg}"
    def warning(msg): return f"⚠️ {msg}"
    def info(msg): return f"ℹ️ {msg}"
    def success(msg): return f"✅ {msg}"


class BaseValidator(ABC):
    """Base class for all step validators with comprehensive error handling and emoji logging."""

    def __init__(self, step_name: str, config: dict[str, Any]) -> None:
        try:
            logger.info(f"🔧 Initializing BaseValidator for step: {step_name}")
            
            # Validate inputs
            if not step_name or not isinstance(step_name, str):
                raise ValueError(f"Step name must be a non-empty string, got: {step_name}")
            
            if not isinstance(config, dict):
                raise ValueError(f"Config must be a dictionary, got: {type(config)}")
            
            self.step_name: str = step_name
            self.config: dict[str, Any] = config
            self.logger = logging.getLogger(f"AresGlobal.{self.__class__.__name__}")
            self.validation_results: dict[str, dict[str, Any]] = {}
            self.validation_count: int = 0
            self.error_count: int = 0
            self.last_validation_time: Optional[str] = None
            
            logger.info(f"✅ BaseValidator initialized successfully for step: {step_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize BaseValidator: {e}")
            log_error_with_context(
                logger, e,
                context={"step_name": step_name, "config_type": type(config).__name__},
                operation="BaseValidator.__init__"
            )
            raise

    def print(self, message: str) -> None:
        """Proxy print to logger to keep output consistent in terminal with emoji enhancement."""
        try:
            # Add emoji prefix based on message content
            if any(word in message.lower() for word in ['error', 'failed', 'exception']):
                enhanced_message = error(message)
            elif any(word in message.lower() for word in ['warning', 'caution', 'alert']):
                enhanced_message = warning(message)
            elif any(word in message.lower() for word in ['success', 'completed', 'passed']):
                enhanced_message = success(message)
            else:
                enhanced_message = info(message)
            
            self.logger.info(enhanced_message)
            
        except Exception as e:
            # Fallback to basic logging if emoji enhancement fails
            self.logger.info(message)
            self.logger.error(f"❌ Error in print method: {e}")

    @abstractmethod
    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate a training step with comprehensive error handling and logging.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        try:
            logger.info(f"🔍 Starting validation for step: {self.step_name}")
            
            # Validate inputs
            if not isinstance(training_input, dict):
                raise ValueError(f"training_input must be a dictionary, got: {type(training_input)}")
            
            if not isinstance(pipeline_state, dict):
                raise ValueError(f"pipeline_state must be a dictionary, got: {type(pipeline_state)}")
            
            # Log validation start
            log_step_progress(
                self.logger, 
                self.step_name, 
                1, 1,  # Single step validation
                "running",
                "Starting validation process"
            )
            
            # Increment validation count
            self.validation_count += 1
            
            # Call the actual validation implementation
            result = await self._perform_validation(training_input, pipeline_state)
            
            # Log validation result
            log_validation_result(
                self.logger,
                self.step_name,
                result,
                f"Validation attempt {self.validation_count}",
                {"validation_count": self.validation_count, "error_count": self.error_count}
            )
            
            # Update last validation time
            from datetime import datetime
            self.last_validation_time = datetime.now().isoformat()
            
            if result:
                logger.success(f"✅ Validation passed for step: {self.step_name}")
            else:
                logger.warning(f"⚠️ Validation failed for step: {self.step_name}")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Validation error for step {self.step_name}: {e}")
            log_error_with_context(
                self.logger, e,
                context={
                    "step_name": self.step_name,
                    "validation_count": self.validation_count,
                    "error_count": self.error_count
                },
                operation=f"BaseValidator.validate({self.step_name})"
            )
            return False

    async def _perform_validation(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Perform the actual validation logic. To be implemented by subclasses.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        raise NotImplementedError("Subclasses must implement _perform_validation method")

    def validate_error_absence(
        self,
        step_result: dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the step completed without errors with comprehensive error handling.

        Args:
            step_result: Step result dictionary

        Returns:
            (passed, metrics)
        """
        try:
            logger.debug(f"🔍 Validating error absence for step: {self.step_name}")
            
            if not isinstance(step_result, dict):
                raise ValueError(f"step_result must be a dictionary, got: {type(step_result)}")
            
            errors = step_result.get("errors", [])
            warnings = step_result.get("warnings", [])

            # Validate error and warning structures
            if not isinstance(errors, list):
                logger.warning(f"⚠️ Errors should be a list, got: {type(errors)}")
                errors = []
            
            if not isinstance(warnings, list):
                logger.warning(f"⚠️ Warnings should be a list, got: {type(warnings)}")
                warnings = []

            critical_errors = [e for e in errors if isinstance(e, dict) and e.get("severity") == "CRITICAL"]

            metrics: dict[str, Any] = {
                "total_errors": len(errors),
                "total_warnings": len(warnings),
                "critical_errors": len(critical_errors),
                "has_critical_errors": len(critical_errors) > 0,
                "error_messages": errors,
                "warning_messages": warnings,
            }

            passed = len(critical_errors) == 0
            
            # Log detailed results
            if passed:
                logger.info(f"✅ No critical errors found in step: {self.step_name}")
                if len(errors) > 0:
                    logger.info(f"ℹ️ Found {len(errors)} non-critical errors")
                if len(warnings) > 0:
                    logger.info(f"ℹ️ Found {len(warnings)} warnings")
            else:
                logger.error(f"❌ Step {self.step_name} has {len(critical_errors)} critical errors")
                for i, error in enumerate(critical_errors):
                    logger.error(f"  {i+1}. {error}")
            
            # Log validation result
            log_validation_result(
                self.logger,
                f"{self.step_name}_error_absence",
                passed,
                f"Critical errors: {len(critical_errors)}, Total errors: {len(errors)}, Warnings: {len(warnings)}",
                metrics
            )

            return passed, metrics

        except Exception as e:
            logger.error(f"❌ Error in error absence validation: {e}")
            log_error_with_context(
                self.logger, e,
                context={"step_name": self.step_name, "step_result_type": type(step_result).__name__},
                operation="BaseValidator.validate_error_absence"
            )
            return False, {"error": str(e)}

    def validate_file_exists(
        self,
        file_path: str,
        file_type: str = "file",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that a file exists with comprehensive error handling.

        Args:
            file_path: Path to the file
            file_type: Type of file for logging

        Returns:
            (passed, metrics)
        """
        try:
            logger.debug(f"🔍 Validating {file_type} existence: {file_path}")
            
            if not file_path or not isinstance(file_path, str):
                raise ValueError(f"file_path must be a non-empty string, got: {file_path}")
            
            if not file_type or not isinstance(file_type, str):
                raise ValueError(f"file_type must be a non-empty string, got: {file_type}")
            
            exists = os.path.exists(file_path)
            
            # Get additional file information if it exists
            file_size = None
            is_file = None
            is_dir = None
            
            if exists:
                try:
                    stat_info = os.stat(file_path)
                    file_size = stat_info.st_size
                    is_file = os.path.isfile(file_path)
                    is_dir = os.path.isdir(file_path)
                except OSError as e:
                    logger.warning(f"⚠️ Could not get file stats: {e}")
            
            metrics: dict[str, Any] = {
                "file_path": file_path,
                "file_type": file_type,
                "exists": exists,
                "file_size": file_size,
                "is_file": is_file,
                "is_directory": is_dir,
            }

            if exists:
                if is_file:
                    logger.info(f"✅ {file_type} found: {file_path} ({file_size} bytes)")
                elif is_dir:
                    logger.warning(f"⚠️ Path exists but is a directory, not a {file_type}: {file_path}")
                    exists = False  # Treat directory as failure for file validation
                else:
                    logger.warning(f"⚠️ Path exists but is neither file nor directory: {file_path}")
                    exists = False
            else:
                logger.error(f"❌ {file_type} not found: {file_path}")
            
            # Log validation result
            log_validation_result(
                self.logger,
                f"{self.step_name}_file_exists",
                exists,
                f"{file_type} at {file_path}",
                metrics
            )

            return exists, metrics
            
        except Exception as e:
            logger.error(f"❌ Error validating {file_type} existence: {e}")
            log_error_with_context(
                self.logger, e,
                context={"file_path": file_path, "file_type": file_type},
                operation="BaseValidator.validate_file_exists"
            )
            return False, {"error": str(e), "file_path": file_path, "file_type": file_type}

    def get_validator_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of this validator.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            logger.info(f"🏥 Getting validator health status for: {self.step_name}")
            
            # Calculate health score
            health_score = 100
            issues = []
            
            if self.error_count > 0:
                health_score -= min(self.error_count * 10, 50)
                issues.append(f"Has {self.error_count} validation errors")
            
            if self.validation_count == 0:
                health_score -= 20
                issues.append("Never validated")
            
            # Check configuration
            if not self.config:
                health_score -= 15
                issues.append("Empty configuration")
            
            # Determine status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 70:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            else:
                status = "poor"
            
            health_info = {
                "step_name": self.step_name,
                "status": status,
                "health_score": health_score,
                "validation_count": self.validation_count,
                "error_count": self.error_count,
                "last_validation_time": self.last_validation_time,
                "config_keys": list(self.config.keys()) if self.config else [],
                "issues": issues,
                "validator_class": self.__class__.__name__
            }
            
            logger.info(f"✅ Validator health check completed: {status} ({health_score}/100)")
            return health_info
            
        except Exception as e:
            logger.error(f"❌ Error getting validator health status: {e}")
            log_error_with_context(
                self.logger, e,
                context={"step_name": self.step_name},
                operation="BaseValidator.get_validator_health_status"
            )
            return {
                "step_name": self.step_name,
                "status": "error",
                "health_score": 0,
                "error": str(e)
            }

    def validate_dataframe_quality(
        self,
        df: pd.DataFrame,
        min_rows: int = 100,
        required_columns: Optional[list[str]] = None,
        check_data_types: bool = True,
        check_value_ranges: bool = True,
        check_duplicates: bool = True,
        check_temporal_consistency: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate DataFrame quality with comprehensive checks.

        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required
            required_columns: List of required columns
            check_data_types: Whether to validate data types
            check_value_ranges: Whether to check for reasonable value ranges
            check_duplicates: Whether to check for duplicate rows
            check_temporal_consistency: Whether to check temporal consistency for time series

        Returns:
            (passed, metrics)
        """
        try:
            metrics: dict[str, Any] = {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "has_minimum_rows": len(df) >= min_rows,
                "missing_columns": [],
                "null_counts": {},
                "data_type_issues": {},
                "value_range_issues": {},
                "duplicate_rows": 0,
                "temporal_issues": {},
                "critical_issues": [],
            }

            # Check minimum rows
            if len(df) < min_rows:
                self.logger.warning(
                    f"⚠️ DataFrame has {len(df)} rows (minimum: {min_rows})",
                )
                metrics["critical_issues"].append(f"Insufficient rows: {len(df)} < {min_rows}")

            # Check required columns
            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                metrics["missing_columns"] = missing_cols
                if missing_cols:
                    self.logger.warning(
                        missing(f"⚠️ Missing required columns: {missing_cols}"),
                    )
                    metrics["critical_issues"].append(f"Missing required columns: {missing_cols}")

            # Check for null values
            for col in df.columns:
                null_count = int(df[col].isnull().sum())
                if null_count > 0:
                    metrics["null_counts"][str(col)] = null_count
                    if null_count > len(df) * 0.1:  # More than 10% nulls
                        metrics["critical_issues"].append(f"High null count in {col}: {null_count}")

            # Check data types
            if check_data_types:
                for col in df.columns:
                    if col in ["open", "high", "low", "close", "volume"]:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            metrics["data_type_issues"][col] = f"Expected numeric, got {df[col].dtype}"
                            metrics["critical_issues"].append(f"Invalid data type for {col}")

            # Check value ranges for financial data
            if check_value_ranges:
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        if (df[col] <= 0).any():
                            negative_count = (df[col] <= 0).sum()
                            metrics["value_range_issues"][col] = f"Negative values: {negative_count}"
                            metrics["critical_issues"].append(f"Negative values in {col}: {negative_count}")

                        # Check OHLC consistency
                        if all(c in df.columns for c in ["open", "high", "low", "close"]):
                            invalid_ohlc = (
                                (df["high"] < df["low"])
                                | (df["high"] < df["open"])
                                | (df["high"] < df["close"])
                                | (df["low"] > df["open"])
                                | (df["low"] > df["close"])
                            ).sum()
                            if invalid_ohlc > 0:
                                metrics["value_range_issues"]["ohlc_consistency"] = f"Invalid OHLC: {invalid_ohlc} rows"
                                metrics["critical_issues"].append(f"OHLC consistency issues: {invalid_ohlc} rows")

            # Check for duplicates
            if check_duplicates:
                duplicate_count = df.duplicated().sum()
                metrics["duplicate_rows"] = duplicate_count
                if duplicate_count > 0:
                    self.logger.warning(f"⚠️ Found {duplicate_count} duplicate rows")
                    if duplicate_count > len(df) * 0.05:  # More than 5% duplicates
                        metrics["critical_issues"].append(f"High duplicate count: {duplicate_count}")

            # Check temporal consistency for time series
            if check_temporal_consistency and isinstance(df.index, pd.DatetimeIndex):
                if len(df) > 1:
                    # Check for gaps in time series
                    time_diff = df.index.to_series().diff().dropna()
                    if len(time_diff) > 0:
                        max_gap = time_diff.max()
                        min_gap = time_diff.min()
                        expected_gap = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else None

                        metrics["temporal_issues"] = {
                            "max_gap": str(max_gap),
                            "min_gap": str(min_gap),
                            "expected_gap": str(expected_gap) if expected_gap else None,
                        }

                        # Check for unusually large gaps
                        if expected_gap and max_gap > expected_gap * 10:
                            metrics["critical_issues"].append(f"Large temporal gap detected: {max_gap}")

            # Determine overall validation result
            passed = (
                len(df) >= min_rows
                and (not required_columns or not metrics["missing_columns"])
                and len(metrics["critical_issues"]) == 0
            )

            return passed, metrics

        except Exception as e:  # pragma: no cover - defensive logging
            self.print(validation_error(f"❌ Error in DataFrame validation: {e}"))
            return False, {"error": str(e)}

    def validate_model_artifacts(
        self,
        model_path: str,
        required_files: Optional[list[str]] = None,
        check_model_integrity: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate model artifacts and integrity.

        Args:
            model_path: Path to model directory or file
            required_files: List of required files in model directory
            check_model_integrity: Whether to perform model integrity checks

        Returns:
            (passed, metrics)
        """
        try:
            metrics: dict[str, Any] = {
                "model_path": model_path,
                "exists": os.path.exists(model_path),
                "is_file": os.path.isfile(model_path) if os.path.exists(model_path) else False,
                "is_directory": os.path.isdir(model_path) if os.path.exists(model_path) else False,
                "file_size": os.path.getsize(model_path) if os.path.isfile(model_path) else 0,
                "missing_files": [],
                "integrity_issues": [],
            }

            if not metrics["exists"]:
                self.logger.warning(missing(f"⚠️ Model path does not exist: {model_path}"))
                return False, metrics

            # Check required files if model is a directory
            if metrics["is_directory"] and required_files:
                for file_name in required_files:
                    file_path = os.path.join(model_path, file_name)
                    if not os.path.exists(file_path):
                        metrics["missing_files"].append(file_name)

            # Check model integrity
            if check_model_integrity and metrics["is_file"]:
                try:
                    import pickle
                    
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                    # Basic model validation
                    if hasattr(model, "predict"):
                        metrics["has_predict_method"] = True
                    else:
                        metrics["integrity_issues"].append("Model missing predict method")

                    if hasattr(model, "fit"):
                        metrics["has_fit_method"] = True
                    else:
                        metrics["integrity_issues"].append("Model missing fit method")

                except Exception as e:
                    metrics["integrity_issues"].append(f"Model loading failed: {str(e)}")

            passed = (
                metrics["exists"]
                and (not required_files or not metrics["missing_files"])
                and (not check_model_integrity or not metrics["integrity_issues"])
            )

            return passed, metrics

        except Exception as e:
            self.print(validation_error(f"❌ Error in model artifacts validation: {e}"))
            return False, {"error": str(e)}

    def validate_configuration(
        self,
        config: dict[str, Any],
        required_keys: Optional[list[str]] = None,
        validate_types: bool = True,
        validate_ranges: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration dictionary to validate
            required_keys: List of required configuration keys
            validate_types: Whether to validate data types
            validate_ranges: Whether to validate value ranges

        Returns:
            (passed, metrics)
        """
        try:
            metrics: dict[str, Any] = {
                "config_keys": list(config.keys()) if isinstance(config, dict) else [],
                "missing_keys": [],
                "type_issues": {},
                "range_issues": {},
                "critical_issues": [],
            }

            if not isinstance(config, dict):
                metrics["critical_issues"].append("Configuration is not a dictionary")
                return False, metrics

            # Check required keys
            if required_keys:
                for key in required_keys:
                    if key not in config:
                        metrics["missing_keys"].append(key)

            # Type validation for common configuration parameters
            if validate_types:
                type_validations = {
                    "symbol": str,
                    "exchange": str,
                    "timeframe": str,
                    "data_dir": str,
                    "min_records": int,
                    "max_gap_ratio": float,
                    "price_tolerance": float,
                }

                for key, expected_type in type_validations.items():
                    if key in config:
                        if not isinstance(config[key], expected_type):
                            metrics["type_issues"][
                                key
                            ] = f"Expected {expected_type.__name__}, got {type(config[key]).__name__}"
                            metrics["critical_issues"].append(f"Invalid type for {key}")

            # Range validation for numeric parameters
            if validate_ranges:
                range_validations = {
                    "min_records": (1, float("inf")),
                    "max_gap_ratio": (0.0, 1.0),
                    "price_tolerance": (0.0, 1.0),
                }

                for key, (min_val, max_val) in range_validations.items():
                    if key in config and isinstance(config[key], (int, float)):
                        if config[key] < min_val or config[key] > max_val:
                            metrics["range_issues"][key] = f"Value {config[key]} outside range [{min_val}, {max_val}]"
                            metrics["critical_issues"].append(f"Invalid range for {key}")

            passed = (
                isinstance(config, dict)
                and (not required_keys or not metrics["missing_keys"])
                and len(metrics["critical_issues"]) == 0
            )

            return passed, metrics

        except Exception as e:
            self.print(validation_error(f"❌ Error in configuration validation: {e}"))
            return False, {"error": str(e)}

    def validate_pipeline_state(
        self,
        pipeline_state: dict[str, Any],
        required_steps: Optional[list[str]] = None,
        check_step_completion: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate pipeline state consistency.

        Args:
            pipeline_state: Pipeline state dictionary
            required_steps: List of required completed steps
            check_step_completion: Whether to check step completion status

        Returns:
            (passed, metrics)
        """
        try:
            metrics: dict[str, Any] = {
                "state_keys": list(pipeline_state.keys()) if isinstance(pipeline_state, dict) else [],
                "missing_steps": [],
                "incomplete_steps": [],
                "failed_steps": [],
                "critical_issues": [],
            }

            if not isinstance(pipeline_state, dict):
                metrics["critical_issues"].append("Pipeline state is not a dictionary")
                return False, metrics

            # Check required steps
            if required_steps:
                for step in required_steps:
                    if step not in pipeline_state:
                        metrics["missing_steps"].append(step)

            # Check step completion status
            if check_step_completion:
                for step_name, step_info in pipeline_state.items():
                    if isinstance(step_info, dict):
                        if step_info.get("status") == "FAILED":
                            metrics["failed_steps"].append(step_name)
                        elif step_info.get("completed") is False:
                            metrics["incomplete_steps"].append(step_name)

            # Check for critical issues
            if metrics["failed_steps"]:
                metrics["critical_issues"].append(f"Failed steps: {metrics['failed_steps']}")

            passed = (
                isinstance(pipeline_state, dict)
                and (not required_steps or not metrics["missing_steps"])
                and len(metrics["critical_issues"]) == 0
            )

            return passed, metrics

        except Exception as e:
            self.print(validation_error(f"❌ Error in pipeline state validation: {e}"))
            return False, {"error": str(e)}

    def validate_directory_structure(
        self,
        directory: str,
        required_files: Optional[list[str]] = None,
        required_dirs: Optional[list[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate directory structure.

        Args:
            directory: Directory to validate
            required_files: List of required files (relative to directory)
            required_dirs: List of required subdirectories

        Returns:
            (passed, metrics)
        """
        try:
            exists = os.path.exists(directory)
            is_directory = os.path.isdir(directory) if exists else False
            metrics: dict[str, Any] = {
                "directory": directory,
                "exists": exists,
                "is_directory": is_directory,
                "missing_files": [],
                "missing_dirs": [],
            }

            # Check if directory exists
            if not exists:
                self.logger.warning(
                    missing(f"⚠️ Directory not found: {directory}"),
                )
                return False, metrics

            # Check if it's actually a directory
            if not is_directory:
                self.logger.warning(
                    f"⚠️ Path exists but is not a directory: {directory}",
                )
                return False, metrics

            # Check required files
            if required_files:
                for file_path in required_files:
                    full_path = os.path.join(directory, file_path)
                    if not os.path.exists(full_path):
                        metrics["missing_files"].append(file_path)
                if metrics["missing_files"]:
                    self.logger.warning(
                        missing(f"⚠️ Missing required files: {metrics['missing_files']}"),
                    )

            # Check required subdirectories
            if required_dirs:
                for subdir in required_dirs:
                    full_path = os.path.join(directory, subdir)
                    if not os.path.exists(full_path) or not os.path.isdir(full_path):
                        metrics["missing_dirs"].append(subdir)
                if metrics["missing_dirs"]:
                    self.logger.warning(
                        missing(f"⚠️ Missing required directories: {metrics['missing_dirs']}"),
                    )

            passed = (
                metrics["exists"]
                and metrics["is_directory"]
                and not metrics["missing_files"]
                and not metrics["missing_dirs"]
            )

            return passed, metrics

        except Exception as e:  # pragma: no cover - defensive logging
            self.print(validation_error(f"❌ Error in directory validation: {e}"))
            return False, {"error": str(e)}

    def log_validation_result(
        self,
        validation_name: str,
        passed: bool,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log validation result.

        Args:
            validation_name: Name of the validation
            passed: Whether validation passed
            metrics: Validation metrics
        """
        if passed:
            self.logger.info(f"✅ {validation_name} validation passed")
        else:
            self.logger.warning(failed(f"❌ {validation_name} validation failed"))

        if metrics:
            self.logger.debug(f"📊 {validation_name} metrics: {metrics}")

    def add_validation_result(
        self,
        validation_name: str,
        passed: bool,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add validation result to the results dictionary.

        Args:
            validation_name: Name of the validation
            passed: Whether validation passed
            metrics: Validation metrics
        """
        self.validation_results[validation_name] = {
            "passed": passed,
            "metrics": metrics or {},
        }

        # Also log the result
        self.log_validation_result(validation_name, passed, metrics)
