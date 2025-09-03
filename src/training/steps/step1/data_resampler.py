"""Data Preparation for Step1_5.

Prepares data for step1_5_data_converter.py processing. This module focuses on:
1. Loading and validating klines data
2. Ensuring data is properly formatted for step1_5 processing
3. Optimizing data storage and access patterns

Note: Actual resampling is handled by step1_5_data_converter.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.logger import system_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    ValidationLevel,
    comprehensive_data_validation,
    copy,
    guard_dataframe_nulls,
    handle_errors,
    import,
    optimize_memory_usage,
    validate_data_quality,
    validate_data_structure,
    with_tracing_span,
)

logger = system_logger.getChild("DataPreparation")


class DataPreparation:
    """Prepares data for step1_5_data_converter.py processing."""

    # Expected data formats for step1_5 processing
    EXPECTED_KLINES_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    EXPECTED_AGGTRADES_COLUMNS = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "timestamp",
        "is_buyer_maker",
    ]

    # Supported timeframes for resampling
    SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    # Timeframe mappings for resampling
    TIMEFRAME_MAPPINGS = {
        "1m": "1T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

    @with_tracing_span("get_klines_files")
    def get_klines_files(
        self, symbol: str, exchange: str, interval: str = "1m"
    ) -> list[Path]:
        """Get all klines files for a symbol and exchange."""
        pattern = f"klines_{exchange}_{symbol}_{interval}_*.csv"
        csv_files = list(self.data_cache_path.glob(pattern))

        # Also get parquet files if they exist
        pattern_parquet = f"klines_{exchange}_{symbol}_{interval}_*.parquet"
        parquet_files = list(self.data_cache_path.glob(pattern_parquet))

        return sorted(csv_files + parquet_files)

    @validate_data_quality()
    @guard_dataframe_nulls(mode="warn", arg_index=0)
    @with_tracing_span("load_klines_data")
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
        default_return=pd.DataFrame(),
        context="data_resampler.load_klines_data"
    )
    def load_klines_data(
        self, symbol: str, exchange: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> pd.DataFrame:
        """Load and combine klines data from multiple files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Combined DataFrame with all klines data

        """
        logger.info(f"📊 Loading klines data for {exchange}_{symbol}")

        klines_files = self.get_klines_files(symbol, exchange)
        logger.info(f"📁 Found {len(klines_files)} klines files")

        if not klines_files:
            logger.warning(f"⚠️ No klines files found for {exchange}_{symbol}")
            return pd.DataFrame()

        # Load all files
        dataframes = []
        for file_path in klines_files:
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path, parse_dates=["timestamp"])
                else:
                    df = pd.read_parquet(file_path)

                # Apply date filters if specified
                if start_date:
                    df = df[df["timestamp"] >= start_date]
                if end_date:
                    df = df[df["timestamp"] <= end_date]

                if len(df) > 0:
                    dataframes.append(df)
                    logger.debug(f"✅ Loaded {file_path.name}: {len(df)} rows")

            except Exception as e:
                logger.exception(f"❌ Error loading {file_path.name}: {e}")
                continue

        if not dataframes:
            logger.warning(f"⚠️ No valid data loaded for {exchange}_{symbol}")
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"]
        )

        logger.info(
            f"📊 Combined klines data: {len(combined_df)} rows from {start_date} to {end_date}",
        )

        return combined_df

    @validate_data_structure
    @with_tracing_span("prepare_for_step1_5")
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
            "ready": False,
            "issues": ["Data preparation failed"],
            "data_summary": {},
        },
        context="data_resampler.prepare_for_step1_5"
    )
    def prepare_for_step1_5(self, symbol: str, exchange: str) -> dict:
        """Prepare data for step1_5_data_converter.py processing.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with preparation results

        """
        logger.info(f"🔧 Preparing data for step1_5 processing: {exchange}_{symbol}")

        preparation_result = {
            "symbol": symbol,
            "exchange": exchange,
            "ready": True,
            "issues": [],
            "data_summary": {},
        }

        # Check klines data availability
        klines_files = self.get_klines_files(symbol, exchange)
        preparation_result["data_summary"]["klines_files"] = len(klines_files)

        if not klines_files:
            preparation_result["ready"] = False
            preparation_result["issues"].append("No klines files found")

        # Validate klines data format
        for file_path in klines_files[:3]:  # Check first 3 files
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path, parse_dates=["timestamp"])
                else:
                    df = pd.read_parquet(file_path)

                # Check columns
                if list(df.columns) != self.EXPECTED_KLINES_COLUMNS:
                    preparation_result["issues"].append(
                        f"Invalid klines format in {file_path.name}",
                    )

                # Check data types
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    preparation_result["issues"].append(
                        f"Invalid timestamp format in {file_path.name}",
                    )

            except Exception as e:
                preparation_result["issues"].append(
                    f"Error reading {file_path.name}: {e}",
                )

        if preparation_result["ready"]:
            logger.info("✅ Data preparation for step1_5 completed successfully")
        else:
            logger.warning("⚠️ Data preparation for step1_5 found issues")
        for issue in preparation_result["issues"]:
            logger.warning(f"  - {issue}")

        return preparation_result

    @optimize_memory_usage
    @with_tracing_span("save_resampled_data")
    @handle_errors(
        exceptions=(
            OSError,
            ValueError,
            TypeError,
            KeyError,
            FileNotFoundError,
            PermissionError,
        ),
        default_return=Path(),
        context="data_resampler.save_resampled_data"
    )
    def save_resampled_data(
        self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str, output_format: str = "parquet"
    ) -> Path:
        """Save resampled data with proper formatting and indexing.

        Args:
            df: Resampled DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (e.g. = '5m', '15m')
            output_format: Output format ('parquet' or 'csv')

        Returns:
            Path to saved file

        """
        if len(df) == 0:
            logger.warning("⚠️ Empty DataFrame provided for saving")
            return None

        # Create output directory (save directly to data_cache for step1_5 compatibility)
        output_dir = self.data_cache_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if output_format.lower() == "parquet":
            filename = f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        else:
            filename = f"klines_{exchange}_{symbol}_{timeframe}_consolidated.csv"

        output_path = output_dir / filename

        # Ensure proper column order and types
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if list(df.columns) != expected_columns:
            if all(col in df.columns for col in expected_columns):
                df = df[expected_columns]
            else:
                logger.error(f"❌ Missing required columns for {timeframe} data")
                return None

        # Ensure proper data types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove any rows with NaN values
        df = df.dropna()

        # Sort by timestamp
        df = df.sort_values("timestamp")

        # Save file
        try:
            if output_format.lower() == "parquet":
                df.to_parquet(output_path, compression="zstd", index=False)
            else:
                df.to_csv(output_path, index=False)

            logger.info(f"✅ Saved {timeframe} data: {output_path} ({len(df)} rows)")
            return output_path

        except Exception as e:
            logger.exception(f"❌ Error saving {timeframe} data: {e}")
            return None

    @optimize_memory_usage
    @with_tracing_span("create_partitioned_dataset")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError),
        default_return=None,
        context="data_resampler.create_partitioned_dataset"
    )
    def create_partitioned_dataset(
        self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str, ) -> Path:
        """Create partitioned Parquet dataset for efficient querying.

        Args:
            df: Resampled DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Path to partitioned dataset

        """
        if len(df) == 0:
            logger.warning("⚠️ Empty DataFrame provided for partitioning")
            return None

        # Create partitioned dataset directory
        dataset_dir = (
            self.data_cache_path / "partitioned" / exchange / symbol / timeframe
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Add partitioning columns
        df_partitioned = df.copy()
        df_partitioned["year"] = df_partitioned["timestamp"].dt.year
        df_partitioned["month"] = df_partitioned["timestamp"].dt.month
        df_partitioned["day"] = df_partitioned["timestamp"].dt.day

        # Save as partitioned dataset
        try:
            df_partitioned.to_parquet(
                dataset_dir,
                partition_cols=["year", "month", "day"],
                compression="zstd",
                index=False
            )

            logger.info(
                f"✅ Created partitioned dataset: {dataset_dir} ({len(df)} rows)",
            )
            return dataset_dir

        except Exception as e:
            logger.exception(f"❌ Error creating partitioned dataset: {e}")
            return None

    @comprehensive_data_validation
    @optimize_memory_usage
    @with_tracing_span("resample_all_timeframes")
    @handle_errors(
        exceptions=(
            OSError,
            ValueError,
            TypeError,
            KeyError,
            pd.errors.EmptyDataError,
            FileNotFoundError,
            PermissionError,
            MemoryError,
        ),
        default_return={
            "symbol": "",
            "exchange": "",
            "timeframes": [],
            "source_rows": 0,
            "resampled_files": {},
            "partitioned_datasets": {},
            "success": False,
            "error": "Resampling failed",
        },
        context="data_resampler.resample_all_timeframes"
    )
    def resample_all_timeframes(
        self, symbol: str, exchange: str, timeframes: list[str] | None = None, start_date: datetime | None = None, end_date: datetime | None = None, create_partitions: bool = True
    ) -> dict:
        """Resample data to all specified timeframes.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to resample to (default: ['5m' = '15m', '30m'])
            start_date: Start date filter
            end_date: End date filter
            create_partitions: Whether to create partitioned datasets

        Returns:
            Dictionary with resampling results

        """
        resampling_start = datetime.now()
        
        if timeframes is None:
            timeframes = ["5m", "15m", "30m"]

        logger.info(f"🔄 RESAMPLING {exchange}_{symbol} TO MULTIPLE TIMEFRAMES")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        logger.info(f"⏰ Target timeframes: {timeframes}")
        logger.info(f"📁 Data cache path: {self.data_cache_path}")
        logger.info(f"🔧 Create partitions: {create_partitions}")
        logger.info("-" * 60)

        # Load source data
        logger.info("📊 LOADING SOURCE KLINES DATA")
        source_df = self.load_klines_data(symbol, exchange, start_date, end_date)

        if len(source_df) == 0:
            logger.error(f"❌ No source data available for {exchange}_{symbol}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframes": timeframes,
                "success": False,
                "error": "No source data available",
            }

        results = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframes": timeframes,
            "source_rows": len(source_df),
            "resampled_files": {},
            "partitioned_datasets": {},
            "success": True,
        }

        # Resample to each timeframe
        for timeframe in timeframes:
            try:
                logger.info(f"🔄 Resampling to {timeframe}...")

                # Resample data
                resampled_df = self.resample_to_timeframe(source_df, timeframe)

                if len(resampled_df) == 0:
                    logger.warning(f"⚠️ No data after resampling to {timeframe}")
                    continue

                # Save resampled data
                output_path = self.save_resampled_data(
                    resampled_df, symbol, exchange, timeframe,
                )

                if output_path:
                    results["resampled_files"][timeframe] = str(output_path)

                # Create partitioned dataset if requested
                if create_partitions:
                    partition_path = self.create_partitioned_dataset(
                        resampled_df, symbol, exchange, timeframe,
                    )
                    if partition_path:
                        results["partitioned_datasets"][timeframe] = str(partition_path)

                logger.info(
                    f"✅ Completed {timeframe} resampling: {len(resampled_df)} rows",
                )

            except Exception as e:
                logger.exception(f"❌ Error resampling to {timeframe}: {e}")
                results["success"] = False
                results["error"] = str(e)
                break

        resampling_end = datetime.now()
        resampling_time = resampling_end - resampling_start
        
        logger.info("-" * 60)
        logger.info("📊 RESAMPLING SUMMARY")
        logger.info(f"⏱️  Total resampling time: {resampling_time}")
        logger.info(f"📊 Source data rows: {results.get('source_rows', 0)}")
        logger.info(f"📁 Resampled files created: {len(results.get('resampled_files', {}))}")
        logger.info(f"📁 Partitioned datasets created: {len(results.get('partitioned_datasets', {}))}")
        logger.info(f"✅ Success: {results.get('success', False)}")
        
        if results.get('resampled_files'):
            logger.info("📊 RESAMPLED FILES CREATED:")
            for timeframe, file_path in results['resampled_files'].items():
                logger.info(f"  • {timeframe}: {file_path}")
        
        if results.get('partitioned_datasets'):
            logger.info("📁 PARTITIONED DATASETS CREATED:")
            for timeframe, dataset_path in results['partitioned_datasets'].items():
                logger.info(f"  • {timeframe}: {dataset_path}")
        
        if results.get('success'):
            logger.info("✅ RESAMPLING COMPLETED SUCCESSFULLY!")
        else:
            logger.error(f"❌ RESAMPLING FAILED: {results.get('error', 'Unknown error')}")
        
        return results

    @validate_data_quality()
    @guard_dataframe_nulls(mode="warn", arg_index=0)
    @with_tracing_span("validate_resampled_data")
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
            "error": "Validation failed",
            "file_path": "",
            "row_count": 0,
            "date_range": {},
            "issues": [],
        },
        context="data_resampler.validate_resampled_data"
    )
    def validate_resampled_data(
        self, symbol: str, exchange: str, timeframe: str, ) -> dict:
        """Validate resampled data quality.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe to validate

        Returns:
            Dictionary with validation results

        """
        logger.info(f"🔍 Validating {timeframe} resampled data for {exchange}_{symbol}")

        # Find resampled file
        output_dir = self.data_cache_path / "resampled" / exchange / symbol
        filename = f"klines_{exchange}_{symbol}_{timeframe}_resampled.parquet"
        file_path = output_dir / filename

        if not file_path.exists():
            return {"valid": False, "error": f"File not found: {file_path}"}

        try:
            # Load and validate data
            df = pd.read_parquet(file_path)

            validation_result = {
                "valid": True,
                "file_path": str(file_path),
                "row_count": len(df),
                "date_range": {
                    "start": df["timestamp"].min(),
                    "end": df["timestamp"].max(),
                },
                "issues": [],
            }

            # Check for required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result["valid"] = False
                validation_result["issues"].append(
                    f"Missing columns: {missing_columns}",
                )

            # Check for null values
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                validation_result["issues"].append(
                    f"Null values found: {null_counts.to_dict()}",
                )

            # Check timestamp ordering
            if not df["timestamp"].is_monotonic_increasing:
                validation_result["issues"].append("Timestamps not in ascending order")

            # Check for price anomalies
            if "high" in df.columns and "low" in df.columns:
                invalid_prices = df[df["high"] < df["low"]]
                if len(invalid_prices) > 0:
                    validation_result["issues"].append(
                        f"Invalid prices: {len(invalid_prices)} rows where high < low",
                    )

            # Check timeframe consistency
            if len(df) > 1:
                time_diffs = df["timestamp"].diff().dropna()
                expected_diff = pd.Timedelta(self.TIMEFRAME_MAPPINGS[timeframe])
                inconsistent_gaps = time_diffs[time_diffs != expected_diff]
                if len(inconsistent_gaps) > 0:
                    validation_result["issues"].append(
                        f"Inconsistent time gaps: {len(inconsistent_gaps)} rows",
                    )

            if validation_result["issues"]:
                validation_result["valid"] = False

            logger.info(
                f"📊 Validation result: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}",
            )
            return validation_result

        except Exception as e:
            return {"valid": False, "error": f"Error reading file: {e}"}

    @validate_data_quality()
    @guard_dataframe_nulls(mode="warn", arg_index=0)
    @with_tracing_span("resample_to_timeframe")
    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError, pd.errors.EmptyDataError),
        default_return=pd.DataFrame(),
        context="data_resampler.resample_to_timeframe"
    )
    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample DataFrame to specified timeframe.

        Args:
            df: Source DataFrame with OHLCV data
            timeframe: Target timeframe (e.g. = '5m', '15m', '30m')

        Returns:
            Resampled DataFrame

        """
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            logger.error(f"❌ Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

        if len(df) == 0:
            logger.warning("⚠️ Empty DataFrame provided for resampling")
            return pd.DataFrame()

        try:
            logger.info(f"🔄 Resampling to {timeframe}...")

            # Ensure timestamp is the index for resampling
            df_resampled = df.copy()
            df_resampled = df_resampled.set_index("timestamp")

            # Get the pandas offset string for the timeframe
            offset_str = self.TIMEFRAME_MAPPINGS[timeframe]

            # Resample OHLCV data
            resampled = df_resampled.resample(offset_str).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                },
            )

            # Remove any periods with no data
            resampled = resampled.dropna()

            # Reset index to get timestamp back as a column
            resampled = resampled.reset_index()

            # Ensure proper column order
            expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            resampled = resampled[expected_columns]

            logger.info(f"✅ Resampled to {timeframe}: {len(resampled)} rows")
            return resampled

        except Exception as e:
            logger.exception(f"❌ Error resampling to {timeframe}: {e}")
            return pd.DataFrame()

    @validate_data_quality()
    @with_tracing_span("validate_resampled_data_quality")
    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={"valid": False, "issues": ["Validation failed"], "warnings": [], "row_count": 0, "timeframe": "unknown"},
        context="data_resampler.validate_resampled_data_quality"
    )
    def validate_resampled_data_quality(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Validate quality of resampled data.

        Args:
            df: Resampled DataFrame
            timeframe: Timeframe of the data

        Returns:
            Dictionary with validation results

        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "row_count": len(df),
            "timeframe": timeframe,
        }

        if len(df) == 0:
            validation_result["valid"] = False
            validation_result["issues"].append("Empty DataFrame")
            return validation_result

        # Check for required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Missing columns: {missing_columns}")

        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            validation_result["issues"].append(
                f"Null values found: {null_counts.to_dict()}",
            )

        # Check timestamp ordering
        if not df["timestamp"].is_monotonic_increasing:
            validation_result["issues"].append("Timestamps not in ascending order")

        # Check for price anomalies
        if "high" in df.columns and "low" in df.columns:
            invalid_prices = df[df["high"] < df["low"]]
            if len(invalid_prices) > 0:
                validation_result["issues"].append(
                    f"Invalid prices: {len(invalid_prices)} rows where high < low",
                )

        # Check timeframe consistency
        if len(df) > 1:
            time_diffs = df["timestamp"].diff().dropna()
            expected_diff = pd.Timedelta(self.TIMEFRAME_MAPPINGS[timeframe])
            inconsistent_gaps = time_diffs[time_diffs != expected_diff]
            if len(inconsistent_gaps) > 0:
                validation_result["warnings"].append(
                    f"Inconsistent time gaps: {len(inconsistent_gaps)} rows",
                )

        # Check for extreme values
        if "volume" in df.columns:
            zero_volume = (df["volume"] == 0).sum()
            if zero_volume > len(df) * 0.1:  # More than 10% zero volume
                validation_result["warnings"].append(
                    f"High number of zero volume periods: {zero_volume}",
                )

        if validation_result["issues"]:
            validation_result["valid"] = False

        return validation_result

    def generate_resampling_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive resampling report."""
        report = f"""
🔄 RESAMPLING REPORT FOR {exchange}_{symbol}
{'='*60}

📊 AVAILABLE TIMEFRAMES:
    pass
"""

        for timeframe in self.SUPPORTED_TIMEFRAMES:
            # Check if resampled file exists
            output_dir = self.data_cache_path / "resampled" / exchange / symbol
            filename = f"klines_{exchange}_{symbol}_{timeframe}_resampled.parquet"
            file_path = output_dir / filename

            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    report += f"• {timeframe}: ✅ Available ({len(df)} rows)\n"
                except:
                    report += f"• {timeframe}: ❌ Corrupted\n"
            else:
                report += f"• {timeframe}: ❌ Not available\n"

        report += f"""
{'='*60}
"""

        return report

    @with_tracing_span("create_1m_consolidated_data")
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
            "success": False,
            "error": "1m consolidation failed",
            "file_path": "",
            "row_count": 0,
        },
        context="data_resampler.create_1m_consolidated_data"
    )
    def create_1m_consolidated_data(self, symbol: str, exchange: str) -> dict:
        """Create 1m consolidated data from klines files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with consolidation results

        """
        logger.info(f"🔧 Creating 1m consolidated data for {exchange}_{symbol}")

        consolidation_result = {
            "symbol": symbol,
            "exchange": exchange,
            "success": False,
            "error": "",
            "file_path": "",
            "row_count": 0,
        }

        try:
            # Load all klines data
            klines_df = self.load_klines_data(symbol, exchange)

            if len(klines_df) == 0:
                consolidation_result["error"] = "No klines data available"
                logger.error("❌ No klines data available for 1m consolidation")
                return consolidation_result

            # Ensure proper column order and types
            expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            if list(klines_df.columns) != expected_columns:
                if all(col in klines_df.columns for col in expected_columns):
                    klines_df = klines_df[expected_columns]
                else:
                    consolidation_result["error"] = "Missing required columns"
                    logger.error("❌ Missing required columns for 1m consolidation")
                    return consolidation_result

            # Ensure proper data types
            klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"])
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                klines_df[col] = pd.to_numeric(klines_df[col], errors="coerce")

            # Remove any rows with NaN values
            klines_df = klines_df.dropna()

            # Sort by timestamp
            klines_df = klines_df.sort_values("timestamp")

            # Remove duplicates
            klines_df = klines_df.drop_duplicates(subset=["timestamp"])

            # Create output path
            output_path = (
                self.data_cache_path
                / f"klines_{exchange}_{symbol}_1m_consolidated.parquet"
            )

            # Save consolidated data
            klines_df.to_parquet(output_path, compression="zstd", index=False)

            consolidation_result["success"] = True
            consolidation_result["file_path"] = str(output_path)
            consolidation_result["row_count"] = len(klines_df)

            logger.info(
                f"✅ Created 1m consolidated data: {output_path} ({len(klines_df)} rows)",
            )

        except Exception as e:
            consolidation_result["error"] = str(e)
            logger.exception(f"❌ Error creating 1m consolidated data: {e}")

        return consolidation_result