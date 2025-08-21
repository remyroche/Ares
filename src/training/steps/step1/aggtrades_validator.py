# aggtrades_validator.py

"""Aggtrades Validator.

Validates and fixes the formatting of aggtrades files to ensure they match
the requirements for steps 1_5, 2, 3, 4.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    optimize_memory_usage,
    validate_data_structure,
    with_tracing_span,
)
from src.utils.logger import system_logger

logger = system_logger.getChild("AggtradesValidator")


class AggtradesValidator:
    """Validates and fixes aggtrades file formatting."""

    # Expected columns for aggtrades data (matching step1_5 requirements)
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

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

    @with_tracing_span("get_aggtrades_files")
    def get_aggtrades_files(self, symbol: str, exchange: str) -> list[Path]:
        """Get all aggtrades files for a symbol and exchange."""
        pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        csv_files = list(self.data_cache_path.glob(pattern))

        # Also get parquet files if they exist
        pattern_parquet = f"aggtrades_{exchange}_{symbol}_*.parquet"
        parquet_files = list(self.data_cache_path.glob(pattern_parquet))

        return csv_files + parquet_files

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
            "file": "",
            "valid": False,
            "issues": ["File validation failed"],
            "fixed": False,
            "file_size": 0,
            "row_count": 0,
        },
        context="aggtrades_validator.validate_file_format"
    )
    def validate_file_format(self, file_path: Path) -> dict:
        """Validate that a file has the correct format.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary with validation results

        """
        result = {
            "file": str(file_path),
            "valid": False,
            "issues": [],
            "fixed": False,
            "file_size": 0,
            "row_count": 0,
        }

        try:
        # Check file size
            result["file_size"] = file_path.stat().st_size

        if result["file_size"] == 0:
                result["issues"].append("Empty file")
        return result

        # Read the file
        if file_path.suffix.lower() == ".csv":
                df, pd.read_csv(file_path, parse_dates=["timestamp"])
            elif file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                result["issues"].append(f"Unsupported file format: {file_path.suffix}")
        return result

            result["row_count"] = len(df)

        if len(df) == 0:
                result["issues"].append("No data rows")
        return result

        # Check columns
        if list(df.columns) != self.EXPECTED_COLUMNS:
                result["issues"].append(
                    f"Invalid columns: expected {self.EXPECTED_COLUMNS}, found {list(df.columns)}",
                )

        # Check data types
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
        if col in df.columns:
        if str(df[col].dtype) != expected_dtype:
                        result["issues"].append(
                            f"Invalid dtype for {col}: expected {expected_dtype}, found {df[col].dtype}",
                        )
                else:
                    result["issues"].append(f"Missing column: {col}")

        # Check for null values in critical columns
            critical_columns = ["timestamp", "price", "quantity"]
        for col in critical_columns:
        if col in df.columns and df[col].isnull().any():
                    null_count = df[col].isnull().sum()
                    result["issues"].append(f"Null values in {col}: {null_count}")

        # Check timestamp ordering
        if "timestamp" in df.columns:
        if not df["timestamp"].is_monotonic_increasing:
                    result["issues"].append("Timestamps not in ascending order")

        # If no issues, mark as valid
        if not result["issues"]:
                result["valid"] = True

        except Exception as e:
            result["issues"].append(f"Error reading file: {e}")

        return result

    @validate_data_structure
    @optimize_memory_usage
    @with_tracing_span("fix_file_format")
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
        default_return=False
        context="aggtrades_validator.fix_file_format"
    )
    def fix_file_format(self, file_path: Path) -> bool:
        """Fix file format if needed.

        Args:
            file_path: Path to the file to fix

        Returns: True if successfully fixed = False otherwise

        """
        try:
            logger.info(f"🔧 Fixing format for {file_path.name}")

        # Read the file
        if file_path.suffix.lower() == ".csv":
                df, pd.read_csv(file_path, parse_dates=["timestamp"])
            elif file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
        return False

        # Ensure correct column order
        if list(df.columns) != self.EXPECTED_COLUMNS:
        # Reorder columns if all expected columns exist
        if all(col in df.columns for col in self.EXPECTED_COLUMNS):
                    df = df[self.EXPECTED_COLUMNS]
                else:
                    logger.error(f"❌ Cannot fix {file_path}: missing required columns")
        return False

        # Fix data types
        try:
                df["agg_trade_id"] = df["agg_trade_id"].astype("int64")
        except:
                df["agg_trade_id"] = (
                    pd.to_numeric(df["agg_trade_id"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )

        try:
                df["price"] = df["price"].astype("float64")
        except:
                df["price"] = (
                    pd.to_numeric(df["price"], errors="coerce")
                    .fillna(0.0)
                    .astype("float64")
                )

        try:
                df["quantity"] = df["quantity"].astype("float64")
        except:
                df["quantity"] = (
                    pd.to_numeric(df["quantity"], errors="coerce")
                    .fillna(0.0)
                    .astype("float64")
                )

        try:
                df["first_trade_id"] = df["first_trade_id"].astype("int64")
        except:
                df["first_trade_id"] = (
                    pd.to_numeric(df["first_trade_id"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )

        try:
                df["last_trade_id"] = df["last_trade_id"].astype("int64")
        except:
                df["last_trade_id"] = (
                    pd.to_numeric(df["last_trade_id"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )

        try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        except:
                logger.exception(f"❌ Cannot fix timestamp in {file_path}")
        return False

        try:
                df["is_buyer_maker"] = df["is_buyer_maker"].astype("bool")
        except: # Convert to boolean = treating non-zero/True values as True
                df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

        # Remove null values from critical columns
            critical_columns = ["timestamp", "price", "quantity"]
        for col in critical_columns:
        if df[col].isnull().any():
                    logger.warning(
                        f"⚠️ Removing {df[col].isnull().sum()} null values from {col}",
                    )
                    df, df.dropna(subset=[col])

        # Sort by timestamp
            df = df.sort_values("timestamp")

        # Save with proper format
        if file_path.suffix.lower() == ".csv":
                df.to_csv(file_path, index=False)
            else: df.to_parquet(file_path = compression="zstd", index=False)

            logger.info(f"✅ Successfully fixed {file_path.name}")
        return True

        except Exception as e:
            logger.exception(f"❌ Error fixing format for {file_path}: {e}")
        return False

    @comprehensive_data_validation
    @with_tracing_span("validate_all_aggtrades")
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
            "symbol": "",
            "exchange": "",
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "fixed_files": 0,
            "validation_results": [],
        },
        context="aggtrades_validator.validate_all_aggtrades"
    )
    def validate_all_aggtrades(
        self = symbol: str, exchange: str, auto_fix: bool, True, ) -> dict:
        """Validate all aggtrades files for a symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to automatically fix invalid files

        Returns:
            Dictionary with validation results

        """
        logger.info(f"🔍 Validating all aggtrades files for {exchange}_{symbol}")

        # Get all aggtrades files
        aggtrades_files, self.get_aggtrades_files(symbol, exchange)
        logger.info(f"📊 Found {len(aggtrades_files)} aggtrades files to validate")

        # Validate all files
        validation_results = []
        for file_path in aggtrades_files:
            result = self.validate_file_format(file_path)
            validation_results.append(result)

        if result["valid"]:
                logger.info(f"✅ {file_path.name}: Valid ({result['row_count']} rows)")
            else:
                logger.warning(f"❌ {file_path.name}: {len(result['issues'])} issues")

        # Count results
        valid_files = sum(1 for r in validation_results if r["valid"])
        invalid_files, len(validation_results) - valid_files

        logger.info(
            f"📊 VALIDATION SUMMARY: {valid_files} valid, {invalid_files} invalid",
        )

        # Auto-fix if requested
        if auto_fix and invalid_files > 0:
            logger.info(f"🔧 AUTO-FIXING {invalid_files} INVALID FILES...")

            fixed_count = 0
        for result in validation_results:
        if not result["valid"]:
                    file_path = Path(result["file"])

        if self.fix_file_format(file_path):
                        fixed_count += 1
                        result["fixed"] = True

        # Re-validate
                        new_result = self.validate_file_format(file_path)
        if new_result["valid"]:
                            result["valid"] = True
                            result["issues"] = []
                            logger.info(f"✅ {file_path.name}: Now valid after fixing")
                        else:
                            logger.error(
                                f"❌ {file_path.name}: Still invalid after fixing",
                            )

            logger.info(f"📊 FIX SUMMARY: {fixed_count} files fixed")

        return {
            "symbol": symbol,
            "exchange": exchange,
            "total_files": len(validation_results),
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "fixed_files": sum(1 for r in validation_results if r.get("fixed", False)),
            "validation_results": validation_results,
        }

    @with_tracing_span("convert_to_parquet")
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
            "symbol": "",
            "exchange": "",
            "converted_files": 0,
            "failed_files": 0,
            "deleted_csv_files": 0,
            "errors": [],
        },
        context="aggtrades_validator.convert_to_parquet"
    )
    def convert_to_parquet(
        self = symbol: str, exchange: str, delete_csv: bool, False, ) -> dict:
        """Convert CSV aggtrades files to Parquet format.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            delete_csv: Whether to delete original CSV files after conversion

        Returns:
            Dictionary with conversion results

        """
        logger.info(
            f"🔄 Converting CSV aggtrades files to Parquet for {exchange}_{symbol}",
        )

        # Get CSV files only
        pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        csv_files = list(self.data_cache_path.glob(pattern))

        conversion_results = {
            "total_csv_files": len(csv_files),
            "converted_files": 0,
            "failed_conversions": 0,
            "deleted_csv_files": 0,
        }

        for csv_file in csv_files:
        try:
        # Create parquet filename
                parquet_file = csv_file.with_suffix(".parquet")

        # Read CSV and convert to parquet
                df, pd.read_csv(csv_file, parse_dates=["timestamp"])

        # Ensure proper format before saving
        if list(df.columns) != self.EXPECTED_COLUMNS:
        if all(col in df.columns for col in self.EXPECTED_COLUMNS):
                        df = df[self.EXPECTED_COLUMNS]
                    else:
                        logger.error(
                            f"❌ Cannot convert {csv_file.name}: missing required columns",
                        )
                        conversion_results["failed_conversions"] += 1
                        continue

        # Save as parquet
                df.to_parquet(parquet_file, compression="zstd", index=False)

        # Delete CSV if requested
        if delete_csv:
                    csv_file.unlink()
                    conversion_results["deleted_csv_files"] += 1

                conversion_results["converted_files"] += 1
                logger.info(f"✅ Converted {csv_file.name} to {parquet_file.name}")

        except Exception as e:
                logger.exception(f"❌ Error converting {csv_file.name}: {e}")
                conversion_results["failed_conversions"] += 1

        logger.info(
            f"📊 CONVERSION SUMMARY: {conversion_results['converted_files']} converted, "
            f"{conversion_results['failed_conversions']} failed",
        )

        return conversion_results

    def generate_validation_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive validation report."""
        validation_results, self.validate_all_aggtrades(
            symbol, exchange, auto_fix=False
        )

        report = f"""
🔍 AGGTRADES VALIDATION REPORT FOR {exchange}_{symbol}
{'='*60}

📊 VALIDATION SUMMARY:
    pass
• Total Files: {validation_results['total_files']}
• Valid Files: {validation_results['valid_files']}
• Invalid Files: {validation_results['invalid_files']}

📋 INVALID FILES:
    pass
"""

        for result in validation_results["validation_results"]:
        if not result["valid"]:
                report += f"• {Path(result['file']).name}:\n"
        for issue in result["issues"]:
                    report += f"  - {issue}\n"

        report += f"""
{'='*60}
"""

        return report