#!/usr/bin/env python3
"""Validate and Fix Aggtrades Format for Step1.

Validates and fixes aggtrades data format to ensure compatibility with all pipeline steps.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.utils.logger import system_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    handle_errors,
    validate_data_quality,
    validate_data_structure,
    with_tracing_span,
)

logger = system_logger.getChild("AggtradesFormatValidator")


class AggtradesFormatValidator:
    """Validates and fixes aggtrades data format for pipeline compatibility."""

    # Expected columns for aggtrades data
    EXPECTED_COLUMNS = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "timestamp",
        "is_buyer_maker",
    ]

    # Expected data types
    EXPECTED_DTYPES = {
        "agg_trade_id": "int64",
        "price": "float64",
        "quantity": "float64",
        "first_trade_id": "int64",
        "last_trade_id": "int64",
        "timestamp": "datetime64[ns]",
        "is_buyer_maker": "bool",
    }

    # Step1_5 specific requirements
    STEP1_5_REQUIREMENTS = {
        "min_timestamp": "2020-01-01",
        "max_timestamp": "2025-12-31",
        "min_rows": 100,
        "max_rows": 10000000,
    }

    # Step2 feature engineering requirements
    STEP2_REQUIREMENTS = {
        "min_price": 0.000001,
        "max_price": 1000000.0,
        "min_quantity": 0.000001,
        "max_quantity": 1000000.0,
    }

    # Step3 regime discovery requirements
    STEP3_REQUIREMENTS = {
        "min_trades_per_day": 100,
        "max_gap_seconds": 3600,  # 1 hour
        "required_time_span_days": 30,
    }

    # Step4 labeling requirements
    STEP4_REQUIREMENTS = {
        "min_labeling_period_hours": 24,
        "max_labeling_period_hours": 168,  # 1 week
        "required_features": ["price", "quantity", "timestamp"],
    }

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

    @with_tracing_span("get_aggtrades_files")
    @validate_data_structure
    @with_tracing_span("validate_file_format")
    @handle_errors(
        exceptions=(
            OSError,
            ValueError,
            TypeError,
            KeyError,
            pd.errors.EmptyDataError,
            FileNotFoundError,
            PermissionError,
            pd.errors.ParserError,
        ),
        default_return={
            "valid": False,
            "issues": ["Validation failed"],
            "warnings": [],
            "file_size": 0,
            "row_count": 0,
            "memory_usage_mb": 0.0,
            "step01_5_compatible": False,
            "step02_compatible": False,
            "step03_compatible": False,
            "step04_compatible": False,
        },
        context="aggtrades_format_validator.validate_file_format"
    )
    def validate_file_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single aggtrades file format for pipeline compatibility.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary with comprehensive validation results

        """
        logger.info(f"🔍 Validating {file_path.name} for pipeline compatibility")

        result = {
            "valid": False,
            "issues": [],
            "warnings": [],
            "file_size": 0,
            "row_count": 0,
            "memory_usage_mb": 0.0,
            "step01_5_compatible": False,
            "step02_compatible": False,
            "step03_compatible": False,
            "step04_compatible": False,
        }

        try:
            # Check file size
            result['file_size'] = file_path.stat().st_size

            if result['file_size'] == 0:
                result['issues'].append("Empty file")
                return result

            # Read the file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                result['issues'].append(f"Unsupported file format: {file_path.suffix}")
                return result

            result['row_count'] = len(df)
            result['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024

            if len(df) == 0:
                result['issues'].append("No data rows")
                return result

            # Step 1: Basic column validation
            if list(df.columns) != self.EXPECTED_COLUMNS:
                result['issues'].append(
                    f"Invalid columns: expected {self.EXPECTED_COLUMNS}, found {list(df.columns)}"
                )

            # Step 2: Data type validation
            for col, expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    if str(df[col].dtype) != expected_dtype:
                        result['issues'].append(
                            f"Invalid dtype for {col}: expected {expected_dtype}, found {df[col].dtype}"
                        )
                else:
                    result['issues'].append(f"Missing column: {col}")

            # Step 3: Step1_5 specific validation
            step01_5_issues = self._validate_step01_5_requirements(df)
            result['issues'].extend(step01_5_issues)

            # Step 4: Step2 compatibility (feature engineering requirements)
            step02_issues = self._validate_step02_compatibility(df)
            result['issues'].extend(step02_issues)

            # Step 5: Step3 compatibility (regime discovery requirements)
            step03_issues = self._validate_step03_compatibility(df)
            result['issues'].extend(step03_issues)

            # Step 6: Step4 compatibility (labeling requirements)
            step04_issues = self._validate_step04_compatibility(df)
            result['issues'].extend(step04_issues)

            # Step 7: Data quality checks
            quality_issues = self._validate_data_quality(df)
            result['issues'].extend(quality_issues)

            # Step 8: Memory optimization warnings
            memory_warnings = self._check_memory_optimization(df)
            result['warnings'].extend(memory_warnings)

            # Determine compatibility
            result['step01_5_compatible'] = len([i for i in result['issues'] if 'step01_5' in i.lower()]) == 0
            result['step02_compatible'] = len([i for i in result['issues'] if 'step2' in i.lower()]) == 0
            result['step03_compatible'] = len([i for i in result['issues'] if 'step3' in i.lower()]) == 0
            result['step04_compatible'] = len([i for i in result['issues'] if 'step4' in i.lower()]) == 0

            # Overall validity
            result['valid'] = len(result['issues']) == 0

        except Exception as e:
            result['issues'].append(f"Error reading file: {e}")

        return result


    def _validate_step01_5_requirements(self, df: pd.DataFrame) -> List[str]:
        """Validate step01_5 specific requirements"""
        issues = []

        if 'timestamp' in df.columns:
            # Check timestamp range
            min_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['min_timestamp'])
            max_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['max_timestamp'])

            if df['timestamp'].min() < min_timestamp:
                issues.append(f"step01_5: Timestamps before {min_timestamp} not supported")

            if df['timestamp'].max() > max_timestamp:
                issues.append(f"step01_5: Timestamps after {max_timestamp} not supported")

            # Check timestamp ordering
            if not df['timestamp'].is_monotonic_increasing:
                issues.append("step01_5: Timestamps not in ascending order")

        # Check row count requirements
        if len(df) < self.STEP1_5_REQUIREMENTS['min_rows']:
            issues.append(f"step01_5: Too few rows ({len(df)} < {self.STEP1_5_REQUIREMENTS['min_rows']})")

        if len(df) > self.STEP1_5_REQUIREMENTS['max_rows']:
            issues.append(f"step01_5: Too many rows ({len(df)} > {self.STEP1_5_REQUIREMENTS['max_rows']})")

        return issues

    def _validate_step02_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step2 feature engineering compatibility"""
        issues = []

        if 'price' in df.columns:
            min_price = df['price'].min()
            max_price = df['price'].max()

            if min_price < self.STEP2_REQUIREMENTS['min_price']:
                issues.append(f"step2: Price too low ({min_price} < {self.STEP2_REQUIREMENTS['min_price']})")

            if max_price > self.STEP2_REQUIREMENTS['max_price']:
                issues.append(f"step2: Price too high ({max_price} > {self.STEP2_REQUIREMENTS['max_price']})")

        if 'quantity' in df.columns:
            min_quantity = df['quantity'].min()
            max_quantity = df['quantity'].max()

            if min_quantity < self.STEP2_REQUIREMENTS['min_quantity']:
                issues.append(f"step2: Quantity too low ({min_quantity} < {self.STEP2_REQUIREMENTS['min_quantity']})")

            if max_quantity > self.STEP2_REQUIREMENTS['max_quantity']:
                issues.append(f"step2: Quantity too high ({max_quantity} > {self.STEP2_REQUIREMENTS['max_quantity']})")

        return issues

    def _validate_step03_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step3 regime discovery compatibility"""
        issues = []

        if 'timestamp' in df.columns:
            # Check time span
            time_span = (df['timestamp'].max() - df['timestamp'].min()).days
            if time_span < self.STEP3_REQUIREMENTS['required_time_span_days']:
                issues.append(f"step3: Insufficient time span ({time_span} days < {self.STEP3_REQUIREMENTS['required_time_span_days']} days)")

            # Check for large gaps
            time_diffs = df['timestamp'].diff().dropna()
            max_gap = time_diffs.max().total_seconds()
            if max_gap > self.STEP3_REQUIREMENTS['max_gap_seconds']:
                issues.append(f"step3: Large time gap detected ({max_gap:.1f}s > {self.STEP3_REQUIREMENTS['max_gap_seconds']}s)")

        return issues

    def _validate_step04_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step4 labeling compatibility"""
        issues = []

        # Check required features
        for feature in self.STEP4_REQUIREMENTS['required_features']:
            if feature not in df.columns:
                issues.append(f"step4: Missing required feature: {feature}")

        if 'timestamp' in df.columns:
            # Check labeling period requirements
            time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600

            if time_span_hours < self.STEP4_REQUIREMENTS['min_labeling_period_hours']:
                issues.append(f"step4: Insufficient labeling period ({time_span_hours:.1f}h < {self.STEP4_REQUIREMENTS['min_labeling_period_hours']}h)")

            if time_span_hours > self.STEP4_REQUIREMENTS['max_labeling_period_hours']:
                issues.append(f"step4: Excessive labeling period ({time_span_hours:.1f}h > {self.STEP4_REQUIREMENTS['max_labeling_period_hours']}h)")

        return issues

    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate general data quality"""
        issues = []

        # Check for null values in critical columns
        critical_columns = ['timestamp', 'price', 'quantity']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                issues.append(f"Data quality: {null_count} null values in {col}")

        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append(f"Data quality: {duplicate_timestamps} duplicate timestamps")

        # Check for negative prices or quantities
        if 'price' in df.columns and (df['price'] < 0).any():
            issues.append("Data quality: Negative prices detected")

        if 'quantity' in df.columns and (df['quantity'] < 0).any():
            issues.append("Data quality: Negative quantities detected")

        return issues

    def _check_memory_optimization(self, df: pd.DataFrame) -> List[str]:
        """Check for memory optimization opportunities"""
        warnings = []

        # Check memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_usage_mb > 100:  # 100 MB threshold
            warnings.append(f"Memory optimization: Large memory usage ({memory_usage_mb:.1f} MB)")

        # Check for inefficient data types
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    warnings.append(f"Memory optimization: {col} has inefficient dtype {actual_dtype} (expected {expected_dtype})")

        return warnings

    @with_tracing_span("fix_file_format")
    @handle_errors(
        exceptions=(
            OSError,
            ValueError,
            TypeError,
            KeyError,
            FileNotFoundError,
            PermissionError,
        ),
        default_return=False,
        context="aggtrades_format_validator.fix_file_format"
    )
    def fix_file_format(self, file_path: Path) -> bool:
        """Fix file format issues to ensure pipeline compatibility.

        Args:
            file_path: Path to the file to fix

        Returns:
            True if successfully fixed, False otherwise

        """
        try:
            logger.info(f"🔧 Fixing format for {file_path.name}")

            # Read the file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
                return False

            # Fix column names if needed
            column_mapping = {
                "a": "agg_trade_id",
                "p": "price",
                "q": "quantity",
                "f": "first_trade_id",
                "l": "last_trade_id",
                "T": "timestamp",
                "m": "is_buyer_maker",
            }

            if list(df.columns) != self.EXPECTED_COLUMNS:
                # Check if we have the old column names
                if all(col in df.columns for col in column_mapping.keys()):
                    df = df.rename(columns=column_mapping)
                else:
                    logger.error(f"❌ Cannot fix column names for {file_path.name}")
                    return False

            # Fix data types
            for col, expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    if expected_dtype == "int64":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif expected_dtype == "float64":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif expected_dtype == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif expected_dtype == "bool":
                        df[col] = df[col].astype(bool)

            # Remove rows with null values in critical columns
            critical_columns = ["timestamp", "price", "quantity"]
            df = df.dropna(subset=critical_columns)

            # Sort by timestamp
            df = df.sort_values("timestamp")

            # Remove duplicates
            df = df.drop_duplicates(subset=["timestamp"])

            # Save the fixed file
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=False)
            else:
                df.to_parquet(file_path, compression="zstd", index=False)

            logger.info(f"✅ Fixed format for {file_path.name}")
            return True

        except Exception as e:
            logger.exception(f"❌ Error fixing {file_path.name}: {e}")
            return False

    @with_tracing_span("validate_all_aggtrades")
    @handle_errors(
        exceptions=(
            OSError,
            ValueError,
            TypeError,
            KeyError,
            FileNotFoundError,
            PermissionError,
        ),
        default_return={
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "fixed_files": 0,
            "errors": [],
        },
        context="aggtrades_format_validator.validate_all_aggtrades"
    )
    def validate_all_aggtrades(
        self, symbol: str, exchange: str, auto_fix: bool = True
    ) -> Dict[str, Any]:
        """Validate all aggtrades files for a symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to automatically fix issues

        Returns:
            Dictionary with validation results

        """
        logger.info(f"🔍 Validating all aggtrades for {exchange}_{symbol}")

        aggtrades_files = self.get_aggtrades_files(symbol, exchange)
        logger.info(f"📁 Found {len(aggtrades_files)} aggtrades files")

        validation_result = {
            "total_files": len(aggtrades_files),
            "valid_files": 0,
            "invalid_files": 0,
            "fixed_files": 0,
            "errors": [],
        }

        for file_path in aggtrades_files:
            try:
                # Validate file format
                validation = self.validate_file_format(file_path)

                if validation["valid"]:
                    validation_result["valid_files"] += 1
                    logger.debug(f"✅ {file_path.name} is valid")
                else:
                    validation_result["invalid_files"] += 1
                    logger.warning(f"⚠️ {file_path.name} has issues: {validation['issues']}")

                    # Auto-fix if enabled
                    if auto_fix:
                        if self.fix_file_format(file_path):
                            validation_result["fixed_files"] += 1
                            logger.info(f"🔧 Fixed {file_path.name}")

            except Exception as e:
                validation_result["errors"].append(f"Error processing {file_path.name}: {e}")
                logger.exception(f"❌ Error processing {file_path.name}: {e}")

        logger.info(
            f"📊 Validation complete: {validation_result['valid_files']} valid, "
            f"{validation_result['invalid_files']} invalid, "
            f"{validation_result['fixed_files']} fixed"
        )

        return validation_result

    @with_tracing_span("generate_compatibility_report")
    def generate_compatibility_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive compatibility report.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Compatibility report string

        """
        aggtrades_files = self.get_aggtrades_files(symbol, exchange)

        report = f"""
🔍 AGGTRADES COMPATIBILITY REPORT FOR {exchange}_{symbol}
{'='*80}

📁 FILES FOUND: {len(aggtrades_files)}

📊 COMPATIBILITY RESULTS:
"""

        total_size = 0
        total_rows = 0
        step01_5_compatible = 0
        step02_compatible = 0
        step03_compatible = 0
        step04_compatible = 0

        for file_path in aggtrades_files:
            try:
                validation = self.validate_file_format(file_path)
                file_size = file_path.stat().st_size
                total_size += file_size

                status = "✅ VALID" if validation["valid"] else "❌ INVALID"
                report += f"• {file_path.name}: {status} ({validation['row_count']} rows, {file_size/1024/1024:.2f} MB)\n"

                if not validation["valid"]:
                    for issue in validation["issues"]:
                        report += f"  - Issue: {issue}\n"

                total_rows += validation["row_count"]
                if validation["step01_5_compatible"]:
                    step01_5_compatible += 1
                if validation["step02_compatible"]:
                    step02_compatible += 1
                if validation["step03_compatible"]:
                    step03_compatible += 1
                if validation["step04_compatible"]:
                    step04_compatible += 1

            except Exception as e:
                report += f"• {file_path.name}: ❌ ERROR ({e})\n"

        report += f"""
📈 SUMMARY:
• Total Files: {len(aggtrades_files)}
• Total Size: {total_size/1024/1024:.2f} MB
• Total Rows: {total_rows:,}

🔧 PIPELINE COMPATIBILITY:
• Step1_5 Compatible: {step01_5_compatible}/{len(aggtrades_files)}
• Step2 Compatible: {step02_compatible}/{len(aggtrades_files)}
• Step3 Compatible: {step03_compatible}/{len(aggtrades_files)}
• Step4 Compatible: {step04_compatible}/{len(aggtrades_files)}

{'='*80}
"""

        return report