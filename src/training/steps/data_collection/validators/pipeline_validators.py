#!/usr/bin/env python3
from src.utils.tprint import tprint

import numpy as np
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

"""
Comprehensive Pipeline Validators for Data Collection

This module provides validators for each step of the data collection pipeline,
ensuring data integrity, quality, and proper flow between steps.
"""

import logging
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.core.domain.decorators import (
    validate_data_quality,
    validate_klines_data_quality,
    ValidationLevel,
    monitor_step_execution,
    ensure_data_integrity
)
import pandas as pd

import datetime
import typing

from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
)

class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"

@dataclass
class ValidationReport:
    """Validation report structure."""
    step_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any]
    timestamp: str
    execution_time: float
    warnings: List[str]
    errors: List[str]

class DataCollectionValidator:
    """Comprehensive validator for data collection pipeline steps."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_reports: List[ValidationReport] = []

    @monitor_step_execution(step_name="data_collection_validation")
    @ensure_data_integrity
    async def validate_step1_data_collection(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs
    ) -> ValidationReport:
        """Validate Step 1: Data Collection."""
        start_time = time.time()
        step_name = "step1_data_collection"

        try:
            self.logger.info('🔍 Starting Step 1 data collection validation...')
            self.logger.info(f'📊 Validation parameters:')
            self.logger.info(f'   📈 Symbol: {symbol}')
            self.logger.info(f'   📊 Exchange: {exchange}')
            self.logger.info(f'   📁 Data directory: {data_dir}')
            self.logger.info(f'   📋 Additional kwargs: {list(kwargs.keys()) if kwargs else "None"}')

            # Check if data directory exists
            self.logger.info('🔍 1. Checking data directory existence...')
            data_path = Path(data_dir)
            if not data_path.exists():
                self.logger.error(f'❌ Data directory does not exist: {data_dir}')
                return ValidationReport(
                    step_name = step_name,
                    result = ValidationResult.FAILED,
                    message = f"Data directory does not exist: {data_dir}",
                    details={"data_dir": data_dir},
                    timestamp = format_datetime(get_current_datetime()),
                    execution_time = time.time() - start_time,
                    warnings=[],
                    errors=[f"Data directory not found: {data_dir}"]
                )

            self.logger.info('✅ Data directory exists')

            # Check for required data files
            self.logger.info('🔍 2. Checking for required data files...')
            required_files = [
                f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"klines_{exchange}_{symbol}_1m.parquet",
                f"volume_{exchange}_{symbol}_consolidated.parquet"
            ]

            self.logger.info(f'📋 Required files: {required_files}')

            missing_files = []
            existing_files = []

            for file_name in required_files:
                file_path = data_path / file_name
                if file_path.exists():
                    existing_files.append(str(file_path))
                    self.logger.info(f'✅ Found: {file_name}')
                else:
                    missing_files.append(file_name)
                    self.logger.warning(f'⚠️ Missing: {file_name}')

            self.logger.info(f'📊 File check summary:')
            self.logger.info(f'   📈 Existing files: {len(existing_files)}/{len(required_files)}')
            self.logger.info(f'   ⚠️ Missing files: {len(missing_files)}')

            # Validate file sizes and basic structure
            self.logger.info('🔍 3. Validating file sizes and structure...')
            file_validations = {}
            warnings = []
            errors = []

            for i, file_path in enumerate(existing_files, 1):
                self.logger.info(f'📊 {i}. Validating file: {Path(file_path).name}')

                try:
                    file_size = Path(file_path).stat().st_size
                    self.logger.info(f'   📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)')

                    if file_size == 0:
                        self.logger.error(f'   ❌ Empty file: {file_path}')
                        errors.append(f"Empty file: {file_path}")
                    elif file_size < 1024:  # Less than 1KB
                        self.logger.warning(f'   ⚠️ Very small file: {file_path} ({file_size} bytes)')
                        warnings.append(f"Very small file: {file_path} ({file_size} bytes)")

                    # Try to read the file to check structure
                    if file_path.endswith('.parquet'):
                        self.logger.info(f'   🔍 Reading parquet file structure...')
                        try:
                            df = standardized_parquet_handler.read_parquet_standardized(file_path)
                            file_validations[file_path] = {
                                "rows": len(df),
                                "columns": list(df.columns),
                                "size_bytes": file_size,
                                "readable": True
                            }

                            self.logger.info(f'   📊 DataFrame info:')
                            self.logger.info(f'      📈 Rows: {len(df):,}')
                            self.logger.info(f'      📊 Columns: {len(df.columns)}')
                            self.logger.info(f'      📋 Column names: {list(df.columns)}')

                            # Basic data quality checks
                            if len(df) == 0:
                                self.logger.error(f'   ❌ Empty DataFrame in {file_path}')
                                errors.append(f"Empty DataFrame in {file_path}")
                            elif len(df) < 100:
                                self.logger.warning(f'   ⚠️ Very few rows in {file_path}: {len(df)}')
                                warnings.append(f"Very few rows in {file_path}: {len(df)}")
                            else:
                                self.logger.info(f'   ✅ DataFrame has sufficient data: {len(df):,} rows')

                        except Exception as e:
                            self.logger.error(f'   ❌ Cannot read parquet file {file_path}: {e}')
                            errors.append(f"Cannot read parquet file {file_path}: {e}")
                            file_validations[file_path] = {
                                "readable": False,
                                "error": str(e)
                            }

                except Exception as e:
                    self.logger.error(f'   ❌ Cannot access file {file_path}: {e}')
                    errors.append(f"Cannot access file {file_path}: {e}")

            # Determine validation result
            self.logger.info('🔍 4. Determining validation result...')
            if errors:
                result = ValidationResult.FAILED
                message = f"Data collection validation failed with {len(errors)} errors"
                self.logger.error(f'❌ Validation FAILED: {len(errors)} errors found')
                for error in errors:
                    self.logger.error(f'   ❌ {error}')
            elif missing_files:
                result = ValidationResult.WARNING
                message = f"Data collection validation passed with {len(missing_files)} missing files"
                warnings.extend([f"Missing file: {f}" for f in missing_files])
                self.logger.warning(f'⚠️ Validation WARNING: {len(missing_files)} missing files')
                for missing_file in missing_files:
                    self.logger.warning(f'   ⚠️ Missing: {missing_file}')
            else:
                result = ValidationResult.PASSED
                message = "Data collection validation passed successfully"
                self.logger.info('✅ Validation PASSED: All checks successful')

            execution_time = time.time() - start_time
            self.logger.info(f'📊 Validation summary:')
            self.logger.info(f'   📊 Result: {result.value}')
            self.logger.info(f'   📊 Execution time: {execution_time:.2f}s')
            self.logger.info(f'   📊 Files found: {len(existing_files)}/{len(required_files)}')
            self.logger.info(f'   📊 Warnings: {len(warnings)}')
            self.logger.info(f'   📊 Errors: {len(errors)}')

            report = ValidationReport(
                step_name = step_name,
                result = result,
                message = message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "data_dir": data_dir,
                    "existing_files": existing_files,
                    "missing_files": missing_files,
                    "file_validations": file_validations,
                    "total_files_found": len(existing_files),
                    "total_files_expected": len(required_files)
                },
                timestamp = format_datetime(get_current_datetime()),
                execution_time = execution_time,
                warnings = warnings,
                errors = errors
            )

            self.validation_reports.append(report)
            return report

        except Exception as e:
            self.logger.exception(f"Error validating {step_name}: {e}")
            return ValidationReport(
                step_name = step_name,
                result = ValidationResult.FAILED,
                message = f"Validation error: {e}",
                details={"error": str(e)},
                timestamp = format_datetime(get_current_datetime()),
                execution_time = time.time() - start_time,
                warnings=[],
                errors=[str(e)]
            )

    @monitor_step_execution(step_name="data_converter_validation")
    @validate_data_quality(
        validation_level = ValidationLevel.WARNING,
        required_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        min_rows = 100,
        max_null_ratio = 0.05,
        check_duplicates = True,
        check_timestamps = True,
        context='data_converter_validation'
    )
    async def validate_step1_5_data_converter(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs
    ) -> ValidationReport:
        """Validate Step 1.5: Data Converter."""
        start_time = time.time()
        step_name = "step1_5_data_converter"

        try:
            self.logger.info(f"🔍 Validating {step_name} for {symbol} on {exchange}")

            # Check for converted data files
            data_path = Path(data_dir)
            converted_file = data_path / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"

            if not converted_file.exists():
                return ValidationReport(
                    step_name = step_name,
                    result = ValidationResult.FAILED,
                    message = f"Converted data file not found: {converted_file}",
                    details={"expected_file": str(converted_file)},
                    timestamp = format_datetime(get_current_datetime()),
                    execution_time = time.time() - start_time,
                    warnings=[],
                    errors=[f"Converted file not found: {converted_file}"]
                )

            # Validate converted data structure and quality
            try:
                df = standardized_parquet_handler.read_parquet_standardized(converted_file)

                # Check required columns
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = set(required_columns) - set(df.columns)

                warnings = []
                errors = []

                if missing_columns:
                    errors.append(f"Missing required columns: {missing_columns}")

                # Check data types
                type_issues = []
                if 'timestamp' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        type_issues.append("timestamp column should be datetime type")

                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            type_issues.append(f"{col} column should be numeric type")

                if type_issues:
                    warnings.extend(type_issues)

                # Check OHLC integrity
                ohlc_issues = []
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    invalid_high = df['high'] < df[['open', 'close']].max(axis = 1)
                    if invalid_high.any():
                        ohlc_issues.append(f"Found {invalid_high.sum()} rows where high < max(open, close)")

                    invalid_low = df['low'] > df[['open', 'close']].min(axis = 1)
                    if invalid_low.any():
                        ohlc_issues.append(f"Found {invalid_low.sum()} rows where low > min(open, close)")

                    invalid_hl = df['high'] < df['low']
                    if invalid_hl.any():
                        ohlc_issues.append(f"Found {invalid_hl.sum()} rows where high < low")

                if ohlc_issues:
                    errors.extend(ohlc_issues)

                # Check for duplicates
                if df.duplicated().any():
                    warnings.append(f"Found {df.duplicated().sum()} duplicate rows")

                # Check timestamp continuity
                if 'timestamp' in df.columns:
                    df_sorted = df.sort_values('timestamp')
                    time_diffs = df_sorted['timestamp'].diff().dropna()
                    if len(time_diffs) > 0:
                        expected_interval = pd.Timedelta(minutes = 1)  # 1-minute intervals
                        irregular_intervals = time_diffs[time_diffs != expected_interval]
                        if len(irregular_intervals) > 0:
                            warnings.append(f"Found {len(irregular_intervals)} irregular time intervals")

                # Check for missing values
                null_counts = df.isnull().sum()
                high_null_cols = null_counts[null_counts > len(df) * 0.05]  # More than 5% nulls
                if not high_null_cols.empty:
                    warnings.append(f"Columns with high null ratios: {high_null_cols.to_dict()}")

                # Determine result
                if errors:
                    result = ValidationResult.FAILED
                    message = f"Data converter validation failed with {len(errors)} errors"
                elif warnings:
                    result = ValidationResult.WARNING
                    message = f"Data converter validation passed with {len(warnings)} warnings"
                else:
                    result = ValidationResult.PASSED
                    message = "Data converter validation passed successfully"

                report = ValidationReport(
                    step_name = step_name,
                    result = result,
                    message = message,
                    details={
                        "symbol": symbol,
                        "exchange": exchange,
                        "converted_file": str(converted_file),
                        "data_shape": df.shape,
                        "columns": list(df.columns),
                        "data_types": df.dtypes.to_dict(),
                        "null_counts": null_counts.to_dict(),
                        "duplicate_count": df.duplicated().sum(),
                        "ohlc_validation": {
                            "high_lt_max_oc": invalid_high.sum() if 'invalid_high' in locals() else 0,
                            "low_gt_min_oc": invalid_low.sum() if 'invalid_low' in locals() else 0,
                            "high_lt_low": invalid_hl.sum() if 'invalid_hl' in locals() else 0
                        }
                    },
                    timestamp = format_datetime(get_current_datetime()),
                    execution_time = time.time() - start_time,
                    warnings = warnings,
                    errors = errors
                )

                self.validation_reports.append(report)
                return report

            except Exception as e:
                return ValidationReport(
                    step_name = step_name,
                    result = ValidationResult.FAILED,
                    message = f"Cannot read converted data file: {e}",
                    details={"file": str(converted_file), "error": str(e)},
                    timestamp = format_datetime(get_current_datetime()),
                    execution_time = time.time() - start_time,
                    warnings=[],
                    errors=[str(e)]
                )

        except Exception as e:
            self.logger.exception(f"Error validating {step_name}: {e}")
            return ValidationReport(
                step_name = step_name,
                result = ValidationResult.FAILED,
                message = f"Validation error: {e}",
                details={"error": str(e)},
                timestamp = format_datetime(get_current_datetime()),
                execution_time = time.time() - start_time,
                warnings=[],
                errors=[str(e)]
            )

    @monitor_step_execution(step_name="data_reading_validation")
    @validate_klines_data_quality(
        required_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        check_ohlc_integrity = True
    )
    async def validate_step2_data_reading(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs
    ) -> ValidationReport:
        """Validate Step 2: Data Reading."""
        start_time = time.time()
        step_name = "step2_data_reading"

        try:
            self.logger.info(f"🔍 Validating {step_name} for {symbol} on {exchange}")

            # Check for processed data files
            data_path = Path(data_dir)
            processed_files = [
                f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"klines_{exchange}_{symbol}_1m.parquet"
            ]

            validation_results = {}
            all_warnings = []
            all_errors = []

            for file_name in processed_files:
                file_path = data_path / file_name

                if not file_path.exists():
                    all_errors.append(f"Required file not found: {file_name}")
                    continue

                try:
                    df = standardized_parquet_handler.read_parquet_standardized(file_path)

                    # Basic structure validation
                    if len(df) == 0:
                        all_errors.append(f"Empty file: {file_name}")
                        continue

                    # Check for required columns
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = set(required_columns) - set(df.columns)

                    if missing_columns:
                        all_errors.append(f"Missing columns in {file_name}: {missing_columns}")
                        continue

                    # Data quality checks
                    quality_issues = []

                    # Check for negative prices
                    price_columns = ['open', 'high', 'low', 'close']
                    for col in price_columns:
                        if col in df.columns:
                            negative_prices = (df[col] <= 0).sum()
                            if negative_prices > 0:
                                quality_issues.append(f"Found {negative_prices} negative/zero prices in {col}")

                    # Check for negative volume
                    if 'volume' in df.columns:
                        negative_volume = (df['volume'] < 0).sum()
                        if negative_volume > 0:
                            quality_issues.append(f"Found {negative_volume} negative volumes")

                    # Check for extreme price movements
                    if all(col in df.columns for col in ['open', 'close']):
                        price_changes = abs(df['close'] - df['open']) / df['open']
                        extreme_moves = (price_changes > 0.5).sum()  # More than 50% change
                        if extreme_moves > 0:
                            quality_issues.append(f"Found {extreme_moves} extreme price movements (>50%)")

                    validation_results[file_name] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "quality_issues": quality_issues,
                        "file_size": file_path.stat().st_size
                    }

                    if quality_issues:
                        all_warnings.extend([f"{file_name}: {issue}" for issue in quality_issues])

                except Exception as e:
                    all_errors.append(f"Cannot read {file_name}: {e}")

            # Determine overall result
            if all_errors:
                result = ValidationResult.FAILED
                message = f"Data reading validation failed with {len(all_errors)} errors"
            elif all_warnings:
                result = ValidationResult.WARNING
                message = f"Data reading validation passed with {len(all_warnings)} warnings"
            else:
                result = ValidationResult.PASSED
                message = "Data reading validation passed successfully"

            report = ValidationReport(
                step_name = step_name,
                result = result,
                message = message,
                details={
                    "symbol": symbol,
                    "exchange": exchange,
                    "data_dir": data_dir,
                    "validation_results": validation_results,
                    "files_checked": len(processed_files)
                },
                timestamp = format_datetime(get_current_datetime()),
                execution_time = time.time() - start_time,
                warnings = all_warnings,
                errors = all_errors
            )

            self.validation_reports.append(report)
            return report

        except Exception as e:
            self.logger.exception(f"Error validating {step_name}: {e}")
            return ValidationReport(
                step_name = step_name,
                result = ValidationResult.FAILED,
                message = f"Validation error: {e}",
                details={"error": str(e)},
                timestamp = format_datetime(get_current_datetime()),
                execution_time = time.time() - start_time,
                warnings=[],
                errors=[str(e)]
            )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation reports."""
        if not self.validation_reports:
            return {"message": "No validation reports available"}

        total_reports = len(self.validation_reports)
        passed = sum(1 for r in self.validation_reports if r.result == ValidationResult.PASSED)
        failed = sum(1 for r in self.validation_reports if r.result == ValidationResult.FAILED)
        warnings = sum(1 for r in self.validation_reports if r.result == ValidationResult.WARNING)

        total_warnings = sum(len(r.warnings) for r in self.validation_reports)
        total_errors = sum(len(r.errors) for r in self.validation_reports)

        return {
            "total_reports": total_reports,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "total_warnings": total_warnings,
            "total_errors": total_errors,
            "success_rate": passed / total_reports if total_reports > 0 else 0,
            "reports": [
                {
                    "step": r.step_name,
                    "result": r.result.value,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "warning_count": len(r.warnings),
                    "error_count": len(r.errors)
                }
                for r in self.validation_reports
            ]
        }

    def print_validation_report(self) -> None:
        """Print a formatted validation report."""
        summary = self.get_validation_summary()

        tprint("\n" + "="*80)
        tprint("📊 DATA COLLECTION PIPELINE VALIDATION REPORT")
        tprint("="*80)
        tprint(f"Total Steps Validated: {summary['total_reports']}")
        tprint(f"✅ Passed: {summary['passed']}")
        tprint(f"❌ Failed: {summary['failed']}")
        tprint(f"⚠️  Warnings: {summary['warnings']}")
        tprint(f"Success Rate: {summary['success_rate']:.1%}")
        tprint(f"Total Warnings: {summary['total_warnings']}")
        tprint(f"Total Errors: {summary['total_errors']}")
        tprint("="*80)

        for report in self.validation_reports:
            status_icon = {
                ValidationResult.PASSED: "✅",
                ValidationResult.FAILED: "❌",
                ValidationResult.WARNING: "⚠️",
                ValidationResult.SKIPPED: "⏭️"
            }[report.result]

            tprint(f"\n{status_icon} {report.step_name}")
            tprint(f"   Result: {report.result.value}")
            tprint(f"   Message: {report.message}")
            tprint(f"   Execution Time: {report.execution_time:.3f}s")

            if report.warnings:
                tprint(f"   Warnings ({len(report.warnings)}):")
                for warning in report.warnings:
                    tprint(f"     • {warning}")

            if report.errors:
                tprint(f"   Errors ({len(report.errors)}):")
                for error in report.errors:
                    tprint(f"     • {error}")

        tprint("="*80)

class PipelineStepValidator:
    """Validator for individual pipeline steps with comprehensive checks."""

    def __init__(self, step_name: str, config: Dict[str, Any]):
        self.step_name = step_name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{step_name}")

    @monitor_step_execution(step_name="pipeline_step_validation")
    async def validate_step_prerequisites(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs
    ) -> ValidationReport:
        """Validate prerequisites for a pipeline step."""
        start_time = time.time()

        try:
            # Check if previous steps have been completed
            # This would be implemented based on the specific step requirements
            prerequisites_met = await self._check_prerequisites(symbol, exchange, data_dir)

            if prerequisites_met:
                result = ValidationResult.PASSED
                message = f"Prerequisites validated for {self.step_name}"
            else:
                result = ValidationResult.FAILED
                message = f"Prerequisites not met for {self.step_name}"

            return ValidationReport(
                step_name = f"{self.step_name}_prerequisites",
                result = result,
                message = message,
                details={"prerequisites_met": prerequisites_met},
                timestamp = format_datetime(get_current_datetime()),
                execution_time = time.time() - start_time,
                warnings=[],
                errors=[]
            )

        except Exception as e:
            self.logger.exception(f"Error validating prerequisites for {self.step_name}: {e}")
            return ValidationReport(
                step_name = f"{self.step_name}_prerequisites",
                result = ValidationResult.FAILED,
                message = f"Prerequisite validation error: {e}",
                details={"error": str(e)},
                timestamp = format_datetime(get_current_datetime()),
                execution_time = time.time() - start_time,
                warnings=[],
                errors=[str(e)]
            )

    async def _check_prerequisites(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """Check if prerequisites are met for the step."""
        try:
            # Check if data directory exists and is accessible
            if not os.path.exists(data_dir):
                self.logger.warning(f"Data directory does not exist: {data_dir}")
                return False
            
            # Check if we have read permissions
            if not os.access(data_dir, os.R_OK):
                self.logger.warning(f"No read access to data directory: {data_dir}")
                return False
            
            # Check if symbol and exchange are valid (non-empty strings)
            if not symbol or not exchange:
                self.logger.warning("Symbol or exchange is empty")
                return False
            
            # Additional checks could be added here based on specific requirements
            # For example: checking for required files, database connections, etc.
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prerequisite check failed: {e}")
            return False

# Export main classes
__all__ = [
    'ValidationResult',
    'ValidationReport',
    'DataCollectionValidator',
    'PipelineStepValidator'
]
