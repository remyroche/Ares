#!/usr / bin / env python3
"""Aggtrades Validator for Step1.

Validates and fixes aggtrades data format for step01_5_data_converter.py processing.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.logger import system_logger

# Add project root to path
import project_root, Path
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
import handle_errors,
    handle_errors,
    optimize_memory_usage,
    validate_data_structure,
    with_tracing_span,
)

logger, system_logger.getChild("AggtradesValidator")

class AggtradesValidator:
    """Validates and fixes aggtrades data format."""

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

    def __init__(self, data_cache_path: str = "data_cache") -> None:
    pass
    pass
        self.data_cache_path, Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok = True)

    @with_tracing_span("get_aggtrades_files")
    def get_aggtrades_files(self, symbol: str, exchange: str) -> list[Path]:
    pass
    pass
        """Get all aggtrades files for a symbol and exchange."""
        pattern, f"aggtrades_{exchange}_{symbol}_*.csv"
        csv_files, list(self.data_cache_path.glob(pattern))

        # Also get parquet files if they exist
        pattern_parquet, f"aggtrades_{exchange}_{symbol}_*.parquet"
        parquet_files, list(self.data_cache_path.glob(pattern_parquet))

        return sorted(csv_files + parquet_files)

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
            "file_size": 0,
            "row_count": 0,
        },
        context="aggtrades_validator.validate_file_format"
    )
    def validate_file_format(self, file_path: Path) -> dict:
    pass
    pass
        """Validate a single aggtrades file format.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary with validation results

        """
        validation_start, datetime.now()
        logger.info(f"🔍 VALIDATING FILE: {file_path.name}")
        logger.info(f"📁 Full path: {file_path}")
        logger.info(f"📊 File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
        logger.info(f"🔍 Validating {file_path.name}")

        result = {
            "valid": False,
            "issues": [],
            "file_size": 0,
            "row_count": 0,
        }

        try:
        # Check file size
            result["file_size"] = file_path.stat().st_size

    except Exception as e:
        pass
    except Exception as e:
        pass
        if result["file_size"] == 0:
    pass
    pass
                result["issues"].append("Empty file")
        return result

        # Read the file
        if file_path.suffix.lower() == ".csv":
    pass
    pass
                df, pd.read_csv(file_path, parse_dates=["timestamp"])
            elif file_path.suffix.lower() == ".parquet":
                df, pd.read_parquet(file_path)
            else:
                result["issues"].append(f"Unsupported file format: {file_path.suffix}")
        return result

            result["row_count"] = len(df)

        if len(df) == 0:
    pass
    pass
                result["issues"].append("No data rows")
        return result

        # Check columns
        if list(df.columns) != self.EXPECTED_COLUMNS:
    pass
    pass
                result["issues"].append(
                    f"Invalid columns: expected {self.EXPECTED_COLUMNS}, found {list(df.columns)}",
                )

        # Check data types
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
    pass
    pass
        if col in df.columns:
    pass
    pass
        if str(df[col].dtype) != expected_dtype:
    pass
    pass
                        result["issues"].append(
                            f"Invalid dtype for {col}: expected {expected_dtype}, found {df[col].dtype}",
                        )
                else:
                    result["issues"].append(f"Missing column: {col}")

        # Check for null values in critical columns
            critical_columns = ["timestamp", "price", "quantity"]
        for col in critical_columns:
    pass
    pass
        if col in df.columns and df[col].isnull().any():
    pass
    pass
                    null_count, df[col].isnull().sum()
                    result["issues"].append(f"Null values in {col}: {null_count}")

        # Check timestamp ordering
        if "timestamp" in df.columns:
    pass
    pass
        if not df["timestamp"].is_monotonic_increasing:
    pass
    pass
                    result["issues"].append("Timestamps not in ascending order")

        # If no issues, mark as valid
        if not result["issues"]:
    pass
    pass
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
        default_return = False,
        context="aggtrades_validator.fix_file_format"
    )
    def fix_file_format(self, file_path: Path) -> bool:
    pass
    pass
        """Fix file format if needed.

        Args:
            file_path: Path to the file to fix

        Returns:
            True if successfully fixed, False otherwise

        """
        try:
            logger.info(f"🔧 Fixing format for {file_path.name}")

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Read the file
        if file_path.suffix.lower() == ".csv":
    pass
    pass
                df, pd.read_csv(file_path, parse_dates=["timestamp"])
            elif file_path.suffix.lower() == ".parquet":
                df, pd.read_parquet(file_path)
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
    pass
    pass
        # Check if we have the old column names
        if all(col in df.columns for col in column_mapping.keys()):
    pass
    pass
                    df, df.rename(columns = column_mapping)
                else:
                    logger.error(f"❌ Cannot fix column names for {file_path.name}")
        return False

        # Fix data types
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
    pass
    pass
        if col in df.columns:
    pass
    pass
        if expected_dtype == "int64":
    pass
    pass
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif expected_dtype == "float64":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif expected_dtype == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif expected_dtype == "bool":
                        df[col] = df[col].astype(bool)

        # Remove rows with null values in critical columns
            critical_columns = ["timestamp", "price", "quantity"]
            df, df.dropna(subset = critical_columns)

        # Sort by timestamp
            df, df.sort_values("timestamp")

        # Remove duplicates
            df, df.drop_duplicates(subset=["timestamp"])

        # Save the fixed file
        if file_path.suffix.lower() == ".csv":
    pass
    pass
                df.to_csv(file_path, index = False)
            else:
                df.to_parquet(file_path, compression="zstd", index = False)

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
        context="aggtrades_validator.validate_all_aggtrades"
    )
    def validate_all_aggtrades(
        self, symbol: str, exchange: str, auto_fix: bool, True
    ) -> dict:
        """Validate all aggtrades files for a symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to automatically fix issues

        Returns:
            Dictionary with validation results

        """
        validation_start, datetime.now()
        logger.info(f"🔍 VALIDATING ALL AGGTRADES FOR {exchange}_{symbol}")
        logger.info(f"🔧 Auto - fix enabled: {auto_fix}")
        logger.info(f"📁 Data cache path: {self.data_cache_path}")
        logger.info("-" * 60)

        aggtrades_files, self.get_aggtrades_files(symbol, exchange)
        logger.info(f"📁 Found {len(aggtrades_files)} aggtrades files to validate")

        # Log file types found
        csv_files = [f for f in aggtrades_files if f.suffix.lower() == ".csv"]
        parquet_files = [f for f in aggtrades_files if f.suffix.lower() == ".parquet"]
        logger.info(f"📊 File types: {len(csv_files)} CSV, {len(parquet_files)} Parquet")

        validation_result = {
            "total_files": len(aggtrades_files),
            "valid_files": 0,
            "invalid_files": 0,
            "fixed_files": 0,
            "errors": [],
        }

        for file_path in aggtrades_files:
    pass
    pass
        try:
        # Validate file format
                validation, self.validate_file_format(file_path)

    except Exception as e:
        pass
    except Exception as e:
        pass
        if validation["valid"]:
    pass
    pass
                    validation_result["valid_files"] += 1
                    logger.debug(f"✅ {file_path.name} is valid")
                else:
                    validation_result["invalid_files"] += 1
                    logger.warning(f"⚠️ {file_path.name} has issues: {validation['issues']}")

        # Auto - fix if enabled
        if auto_fix:
    pass
    pass
        if self.fix_file_format(file_path):
    pass
    pass
                            validation_result["fixed_files"] += 1
                            logger.info(f"🔧 Fixed {file_path.name}")

        except Exception as e:
                validation_result["errors"].append(f"Error processing {file_path.name}: {e}")
                logger.exception(f"❌ Error processing {file_path.name}: {e}")

        validation_end, datetime.now()
        validation_time, validation_end - validation_start

        logger.info("-" * 60)
        logger.info("📊 AGGTRADES VALIDATION SUMMARY")
        logger.info(f"⏱️  Validation time: {validation_time}")
        logger.info(f"📁 Total files processed: {validation_result['total_files']}")
        logger.info(f"✅ Valid files: {validation_result['valid_files']}")
        logger.info(f"❌ Invalid files: {validation_result['invalid_files']}")
        logger.info(f"🔧 Fixed files: {validation_result['fixed_files']}")
        logger.info(f"📊 Success rate: {validation_result['valid_files']/validation_result['total_files']*100:.1f}%" if validation_result['total_files'] > 0 else "📊 Success rate: N / A")

        if validation_result['errors']:
    pass
    pass
            logger.error("❌ VALIDATION ERRORS:")
        for i, error in enumerate(validation_result['errors'], 1):
    pass
    pass
                logger.error(f"  {i}. {error}")

        if validation_result['invalid_files'] > 0 and not auto_fix:
    pass
    pass
            logger.warning("⚠️  Some files are invalid and auto - fix is disabled!")
        elif validation_result['invalid_files'] == 0:
            logger.info("✅ All aggtrades files are valid!")

        return validation_result

    @with_tracing_span("convert_to_parquet")
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
            "converted_files": 0,
            "failed_files": 0,
            "errors": [],
        },
        context="aggtrades_validator.convert_to_parquet"
    )
    def convert_to_parquet(self, symbol: str, exchange: str) -> dict:
    pass
    pass
        """Convert CSV aggtrades files to parquet format.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with conversion results

        """
        logger.info(f"🔄 Converting aggtrades to parquet for {exchange}_{symbol}")

        aggtrades_files, self.get_aggtrades_files(symbol, exchange)
        csv_files = [f for f in aggtrades_files if f.suffix.lower() == ".csv"]

        conversion_result = {
            "converted_files": 0,
            "failed_files": 0,
            "errors": [],
        }

        for csv_file in csv_files:
    pass
    pass
        try:
        # Read CSV file
                df, pd.read_csv(csv_file, parse_dates=["timestamp"])

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Create parquet file path
                parquet_file, csv_file.with_suffix(".parquet")

        # Save as parquet
                df.to_parquet(parquet_file, compression="zstd", index = False)

        # Remove original CSV file
                csv_file.unlink()

                conversion_result["converted_files"] += 1
                logger.info(f"✅ Converted {csv_file.name} to parquet")

        except Exception as e:
                conversion_result["failed_files"] += 1
                conversion_result["errors"].append(f"Error converting {csv_file.name}: {e}")
                logger.exception(f"❌ Error converting {csv_file.name}: {e}")

        logger.info(
            f"📊 Conversion complete: {conversion_result['converted_files']} converted, "
            f"{conversion_result['failed_files']} failed"
        )

        return conversion_result

    @with_tracing_span("generate_validation_report")
    def generate_validation_report(self, symbol: str, exchange: str) -> str:
    pass
    pass
        """Generate a validation report for aggtrades files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Validation report string

        """
        aggtrades_files, self.get_aggtrades_files(symbol, exchange)

        report, f"""
🔍 AGGTRADES VALIDATION REPORT FOR {exchange}_{symbol}
{'='*60}

📁 FILES FOUND: {len(aggtrades_files)}

📊 VALIDATION RESULTS:
    pass
"""

        total_size, 0
        total_rows, 0

        for file_path in aggtrades_files:
    pass
    pass
        try:
                validation, self.validate_file_format(file_path)
    except Exception as e:
        pass
    except Exception as e:
        pass
                file_size, file_path.stat().st_size
                total_size += file_size

                status = "✅ VALID" if validation["valid"] else "❌ INVALID"
                report += f"• {file_path.name}: {status} ({validation['row_count']} rows, {file_size / 1024 / 1024:.2f} MB)\\\n"

        if not validation["valid"]:
    pass
    pass
        for issue in validation["issues"]:
    pass
    pass
                        report += f"  - Issue: {issue}\\\n"

                total_rows += validation["row_count"]

        except Exception as e:
                report += f"• {file_path.name}: ❌ ERROR ({e})\\\n"

        report += f"""
📈 SUMMARY:
    pass
• Total Files: {len(aggtrades_files)}
• Total Size: {total_size / 1024 / 1024:.2f} MB
• Total Rows: {total_rows:,}
{'='*60}
"""

        return report