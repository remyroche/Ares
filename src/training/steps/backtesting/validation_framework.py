#!/usr/bin/env python3
"""
Backtesting Validation Framework

This module provides comprehensive validation for the backtesting pipeline,
ensuring data quality, format consistency, and operational integrity at each step.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
)
from src.core.domain.decorators import validate_data_quality, ValidationLevel
from src.utils.compat import handle_errors


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = format_datetime(get_current_datetime())


class BacktestingValidator:
    """Comprehensive validator for backtesting pipeline operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []
        
    def add_result(self, result: ValidationResult):
        """Add a validation result to the collection."""
        self.validation_results.append(result)
        self.logger.info(f"Validation {result.status}: {result.message}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.validation_results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in self.validation_results if r.status == ValidationStatus.WARNING)
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "success_rate": passed / total if total > 0 else 0.0,
            "results": self.validation_results
        }


class DataFormatValidator(BacktestingValidator):
    """Validator for data formatting operations."""
    
    @validate_data_quality(
        validation_level=ValidationLevel.ERROR,
        required_columns=["timestamp", "open", "high", "low", "close", "volume"],
        min_rows=100,
        max_null_ratio=0.05,
        check_duplicates=True,
        check_timestamps=True,
        check_nan=True,
        check_infinite=True
    )
    def validate_price_data(self, data: pd.DataFrame, symbol: str, exchange: str) -> ValidationResult:
        """Validate price data format and quality."""
        try:
            issues = []
            warnings = []
            
            # Check required columns
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
            
            # Check data types
            if "timestamp" in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                    issues.append("Timestamp column must be datetime type")
            
            # Check price data consistency
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        issues.append(f"Non-positive values found in {col} column")
                    if data[col].isnull().any():
                        issues.append(f"Null values found in {col} column")
            
            # Check OHLC consistency
            if all(col in data.columns for col in price_cols):
                invalid_ohlc = (
                    (data["high"] < data["low"]) |
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"]) |
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"])
                )
                if invalid_ohlc.any():
                    issues.append(f"Invalid OHLC relationships found in {invalid_ohlc.sum()} rows")
            
            # Check volume data
            if "volume" in data.columns:
                if (data["volume"] < 0).any():
                    issues.append("Negative volume values found")
                if data["volume"].isnull().any():
                    issues.append("Null values found in volume column")
            
            # Check for gaps in timestamps
            if "timestamp" in data.columns and len(data) > 1:
                time_diff = data["timestamp"].diff().dropna()
                expected_interval = time_diff.mode().iloc[0] if len(time_diff) > 0 else None
                if expected_interval:
                    large_gaps = time_diff > expected_interval * 2
                    if large_gaps.any():
                        warnings.append(f"Large time gaps detected in {large_gaps.sum()} locations")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Data validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Data validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Data validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "data_shape": data.shape,
                    "date_range": {
                        "start": data["timestamp"].min().isoformat() if "timestamp" in data.columns else None,
                        "end": data["timestamp"].max().isoformat() if "timestamp" in data.columns else None
                    }
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Data validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result
    
    def validate_feature_data(self, features: pd.DataFrame, symbol: str, exchange: str) -> ValidationResult:
        """Validate feature engineering output."""
        try:
            issues = []
            warnings = []
            
            # Check for required feature columns
            if len(features.columns) == 0:
                issues.append("No feature columns found")
            
            # Check for infinite values
            numeric_features = features.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                inf_count = np.isinf(numeric_features).sum().sum()
                if inf_count > 0:
                    issues.append(f"Found {inf_count} infinite values in features")
            
            # Check for excessive null values
            null_ratio = features.isnull().sum() / len(features)
            high_null_cols = null_ratio[null_ratio > 0.5]
            if not high_null_cols.empty:
                warnings.append(f"High null ratio in columns: {high_null_cols.to_dict()}")
            
            # Check for constant features
            constant_features = []
            for col in numeric_features.columns:
                if numeric_features[col].nunique() <= 1:
                    constant_features.append(col)
            if constant_features:
                warnings.append(f"Constant features detected: {constant_features}")
            
            # Check feature correlation
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                high_corr_pairs = []
                for col in upper_triangle.columns:
                    high_corr_cols = upper_triangle.index[upper_triangle[col] > 0.95].tolist()
                    if high_corr_cols:
                        high_corr_pairs.extend([(col, c) for c in high_corr_cols])
                
                if high_corr_pairs:
                    warnings.append(f"High correlation detected between features: {high_corr_pairs[:5]}")  # Show first 5
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Feature validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Feature validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Feature validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "feature_count": len(features.columns),
                    "feature_shape": features.shape,
                    "numeric_features": len(numeric_features.columns)
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Feature validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class DataAccessValidator(BacktestingValidator):
    """Validator for data access operations."""
    
    def validate_file_access(self, file_path: Union[str, Path], operation: str = "read") -> ValidationResult:
        """Validate file access permissions and existence."""
        try:
            file_path = Path(file_path)
            issues = []
            warnings = []
            
            # Check file existence
            if not file_path.exists():
                issues.append(f"File does not exist: {file_path}")
                status = ValidationStatus.FAILED
                message = f"File access validation failed: {file_path} not found"
            else:
                # Check file permissions
                if operation == "read" and not file_path.is_file():
                    issues.append(f"Path is not a file: {file_path}")
                elif operation == "write" and not file_path.parent.exists():
                    issues.append(f"Parent directory does not exist: {file_path.parent}")
                
                # Check file size
                file_size = file_path.stat().st_size
                if file_size == 0:
                    warnings.append(f"File is empty: {file_path}")
                elif file_size > 1024 * 1024 * 1024:  # 1GB
                    warnings.append(f"Large file detected: {file_size / (1024**3):.2f}GB")
                
                # Check file extension
                valid_extensions = ['.parquet', '.csv', '.json', '.pkl', '.h5']
                if file_path.suffix.lower() not in valid_extensions:
                    warnings.append(f"Unusual file extension: {file_path.suffix}")
                
                if issues:
                    status = ValidationStatus.FAILED
                    message = f"File access validation failed for {file_path}"
                elif warnings:
                    status = ValidationStatus.WARNING
                    message = f"File access validation passed with warnings for {file_path}"
                else:
                    status = ValidationStatus.PASSED
                    message = f"File access validation passed for {file_path}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "file_path": str(file_path),
                    "operation": operation,
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                    "file_extension": file_path.suffix
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"File access validation error for {file_path}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result
    
    def validate_directory_access(self, dir_path: Union[str, Path], operation: str = "read") -> ValidationResult:
        """Validate directory access permissions."""
        try:
            dir_path = Path(dir_path)
            issues = []
            warnings = []
            
            # Check directory existence
            if not dir_path.exists():
                if operation == "write":
                    # Try to create directory
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        warnings.append(f"Created directory: {dir_path}")
                    except Exception as e:
                        issues.append(f"Cannot create directory {dir_path}: {str(e)}")
                else:
                    issues.append(f"Directory does not exist: {dir_path}")
            
            if not issues:
                # Check permissions
                if operation == "read" and not dir_path.is_dir():
                    issues.append(f"Path is not a directory: {dir_path}")
                elif operation == "write" and not os.access(dir_path, os.W_OK):
                    issues.append(f"No write permission for directory: {dir_path}")
                
                # Check directory contents
                try:
                    contents = list(dir_path.iterdir())
                    if len(contents) == 0:
                        warnings.append(f"Directory is empty: {dir_path}")
                except Exception as e:
                    warnings.append(f"Cannot list directory contents: {str(e)}")
            
            if issues:
                status = ValidationStatus.FAILED
                message = f"Directory access validation failed for {dir_path}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Directory access validation passed with warnings for {dir_path}"
            else:
                status = ValidationStatus.PASSED
                message = f"Directory access validation passed for {dir_path}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "directory_path": str(dir_path),
                    "operation": operation,
                    "exists": dir_path.exists(),
                    "is_directory": dir_path.is_dir() if dir_path.exists() else False
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Directory access validation error for {dir_path}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class AnalysisValidator(BacktestingValidator):
    """Validator for analysis operations."""
    
    def validate_backtest_results(self, results: Dict[str, Any], symbol: str, exchange: str) -> ValidationResult:
        """Validate backtesting results."""
        try:
            issues = []
            warnings = []
            
            # Check required result fields
            required_fields = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
            missing_fields = set(required_fields) - set(results.keys())
            if missing_fields:
                issues.append(f"Missing required result fields: {missing_fields}")
            
            # Validate numeric results
            for field in required_fields:
                if field in results:
                    value = results[field]
                    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                        issues.append(f"Invalid value for {field}: {value}")
            
            # Check for reasonable ranges
            if "total_return" in results:
                if results["total_return"] > 10.0:  # 1000% return
                    warnings.append(f"Unusually high total return: {results['total_return']:.2%}")
                elif results["total_return"] < -0.9:  # -90% return
                    warnings.append(f"Unusually low total return: {results['total_return']:.2%}")
            
            if "sharpe_ratio" in results:
                if results["sharpe_ratio"] > 5.0:
                    warnings.append(f"Unusually high Sharpe ratio: {results['sharpe_ratio']:.2f}")
                elif results["sharpe_ratio"] < -2.0:
                    warnings.append(f"Unusually low Sharpe ratio: {results['sharpe_ratio']:.2f}")
            
            if "max_drawdown" in results:
                if results["max_drawdown"] > 0.5:  # 50% drawdown
                    warnings.append(f"High maximum drawdown: {results['max_drawdown']:.2%}")
            
            if "win_rate" in results:
                if results["win_rate"] > 0.8:  # 80% win rate
                    warnings.append(f"Unusually high win rate: {results['win_rate']:.2%}")
                elif results["win_rate"] < 0.2:  # 20% win rate
                    warnings.append(f"Unusually low win rate: {results['win_rate']:.2%}")
            
            # Check for signal count
            if "signal_count" in results:
                if results["signal_count"] == 0:
                    warnings.append("No trading signals generated")
                elif results["signal_count"] < 10:
                    warnings.append(f"Very few trading signals: {results['signal_count']}")
            
            # Determine status
            if issues:
                status = ValidationStatus.FAILED
                message = f"Backtest results validation failed for {symbol} on {exchange}"
            elif warnings:
                status = ValidationStatus.WARNING
                message = f"Backtest results validation passed with warnings for {symbol} on {exchange}"
            else:
                status = ValidationStatus.PASSED
                message = f"Backtest results validation passed for {symbol} on {exchange}"
            
            result = ValidationResult(
                status=status,
                message=message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "result_fields": list(results.keys()),
                    "validation_timestamp": format_datetime(get_current_datetime())
                },
                warnings=warnings,
                errors=issues
            )
            
            self.add_result(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Backtest results validation error for {symbol} on {exchange}: {str(e)}",
                errors=[str(e)]
            )
            self.add_result(result)
            return result


class BacktestingValidationOrchestrator:
    """Orchestrates all validation operations for the backtesting pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_format_validator = DataFormatValidator(config)
        self.data_access_validator = DataAccessValidator(config)
        self.analysis_validator = AnalysisValidator(config)
        self.all_validators = [
            self.data_format_validator,
            self.data_access_validator,
            self.analysis_validator
        ]
    
    async def validate_pipeline_step(
        self,
        step_name: str,
        data: Optional[pd.DataFrame] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        results: Optional[Dict[str, Any]] = None,
        symbol: str = "ETHUSDT",
        exchange: str = "BINANCE"
    ) -> ValidationResult:
        """Validate a complete pipeline step."""
        try:
            self.logger.info(f"Starting validation for step: {step_name}")
            step_results = []
            
            # Validate data if provided
            if data is not None:
                if step_name in ["data_loading", "data_preprocessing"]:
                    result = self.data_format_validator.validate_price_data(data, symbol, exchange)
                elif step_name in ["feature_engineering", "feature_selection"]:
                    result = self.data_format_validator.validate_feature_data(data, symbol, exchange)
                else:
                    result = self.data_format_validator.validate_price_data(data, symbol, exchange)
                step_results.append(result)
            
            # Validate file access if file paths provided
            if file_paths:
                for file_path in file_paths:
                    result = self.data_access_validator.validate_file_access(file_path)
                    step_results.append(result)
            
            # Validate results if provided
            if results is not None:
                result = self.analysis_validator.validate_backtest_results(results, symbol, exchange)
                step_results.append(result)
            
            # Determine overall step status
            if not step_results:
                overall_result = ValidationResult(
                    status=ValidationStatus.SKIPPED,
                    message=f"No validation performed for step: {step_name}"
                )
            else:
                failed_results = [r for r in step_results if r.status == ValidationStatus.FAILED]
                warning_results = [r for r in step_results if r.status == ValidationStatus.WARNING]
                
                if failed_results:
                    overall_result = ValidationResult(
                        status=ValidationStatus.FAILED,
                        message=f"Step validation failed: {step_name}",
                        details={"step_name": step_name, "failed_validations": len(failed_results)},
                        errors=[r.message for r in failed_results]
                    )
                elif warning_results:
                    overall_result = ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"Step validation passed with warnings: {step_name}",
                        details={"step_name": step_name, "warning_validations": len(warning_results)},
                        warnings=[r.message for r in warning_results]
                    )
                else:
                    overall_result = ValidationResult(
                        status=ValidationStatus.PASSED,
                        message=f"Step validation passed: {step_name}",
                        details={"step_name": step_name, "total_validations": len(step_results)}
                    )
            
            self.logger.info(f"Step validation completed: {overall_result.status} - {overall_result.message}")
            return overall_result
            
        except Exception as e:
            self.logger.exception(f"Error in pipeline step validation: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Step validation error: {step_name} - {str(e)}",
                errors=[str(e)]
            )
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all validations."""
        summary = {
            "overall_summary": {},
            "validator_summaries": {},
            "timestamp": format_datetime(get_current_datetime())
        }
        
        # Overall summary
        all_results = []
        for validator in self.all_validators:
            all_results.extend(validator.validation_results)
        
        if all_results:
            total = len(all_results)
            passed = sum(1 for r in all_results if r.status == ValidationStatus.PASSED)
            failed = sum(1 for r in all_results if r.status == ValidationStatus.FAILED)
            warnings = sum(1 for r in all_results if r.status == ValidationStatus.WARNING)
            
            summary["overall_summary"] = {
                "total_validations": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": passed / total if total > 0 else 0.0
            }
        
        # Individual validator summaries
        for validator in self.all_validators:
            validator_name = validator.__class__.__name__
            summary["validator_summaries"][validator_name] = validator.get_summary()
        
        return summary
    
    def save_validation_report(self, output_path: Union[str, Path]) -> bool:
        """Save a comprehensive validation report."""
        try:
            output_path = Path(output_path)
            ensure_directory(output_path.parent)
            
            report = self.get_comprehensive_summary()
            safe_json_dump(report, output_path, indent=2)
            
            self.logger.info(f"Validation report saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Failed to save validation report: {e}")
            return False