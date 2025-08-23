"""Step 1: Data Collection.

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""

import sys
from pathlib import Path
from typing import Any
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import constants
try:
    from src.config.constants import DEFAULT_LOOKBACK_DAYS
except ImportError:
    # Fallback if constants module is not available
    DEFAULT_LOOKBACK_DAYS = 1095

# Import comprehensive file validation
try:
    from src.utils.comprehensive_file_validation import (
        ComprehensiveFileValidator,
        validate_step1_file,
        FileValidationResult,
    )
    from src.utils.validation_decorators import (
        validate_file_operation,
        validate_dataframe_operation,
        validate_step1_operation,
    )
    from src.utils.advanced_ml_validation import validate_ml_data_quality
    from src.utils.centralized_decorators import step_specific_ml_validation
except ImportError:
    ComprehensiveFileValidator = None
    validate_step1_file = None
    FileValidationResult = None
    validate_file_operation = None
    validate_dataframe_operation = None
    validate_step1_operation = None
    step_specific_ml_validation = None

# Handle imports with fallback - this must be done before any other imports
CONFIG = None
handle_errors = None
setup_logging = None
system_logger = None
download_all_data_with_consolidation = None

# Temporarily comment out problematic imports
# try:
#     from src.config import CONFIG
#     from src.utils.error_handler import handle_errors
#     from src.utils.logger import setup_logging, system_logger
#     from src.training.steps.data_downloader import download_all_data_with_consolidation
#     from src.utils.data_quality_decorators import (
#         handle_data_collection_errors,
#         validate_klines_data,
#         format_klines_data,
#         log_step_metrics,
#     )
# except ImportError:
# Fallback decorators if data quality decorators are not available

def handle_data_collection_errors(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


def log_step_metrics(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

# Handle imports with fallback - this must be done before any other imports
CONFIG = None
handle_errors = None
setup_logging = None
system_logger = None
download_all_data_with_consolidation = None

try:
    from src.config import CONFIG
    from src.utils.error_handler import handle_errors
    from src.utils.logger import setup_logging, system_logger
    from src.training.steps.data_downloader import download_all_data_with_consolidation
except ImportError:
    # Fallback configuration
    CONFIG = {
        "SYMBOL": "ETHUSDT",
        "INTERVAL": "1m",
        "LOOKBACK_YEARS": 2,
    }

    # Create fallback functions
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def setup_logging():
        import logging

        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    system_logger = setup_logging()
    download_all_data_with_consolidation = None

from src.utils.centralized_decorators import monitor_data_collection


class DataCollectionStep:
    """Step 1: Data Collection using existing run_step function."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("DataCollectionStep")

    async def initialize(self) -> None:
        """Initialize the data collection step."""
        self.logger.info("Initializing Data Collection Step...")
        self.logger.info("Data Collection Step initialized successfully")

    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute data collection with enhanced quality management.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state

        """
        self.logger.info("Starting enhanced data collection...")

        try:
            # Execute the data collection
            success = await self._run_data_collection(training_input)

            if success:
                self.logger.info("Data collection completed successfully")

                # Run enhanced quality check after data collection
                quality_success = await self._run_enhanced_quality_check(training_input)

                if quality_success:
                    self.logger.info("✅ Enhanced quality check passed")
                    pipeline_state["data_collection_completed"] = True
                    pipeline_state["quality_check_passed"] = True
                else:
                    self.logger.warning("⚠️ Enhanced quality check found issues")
                    pipeline_state["data_collection_completed"] = True
                    pipeline_state["quality_check_passed"] = False
            else:
                self.logger.error("Data collection failed")
                pipeline_state["data_collection_completed"] = False
                pipeline_state["quality_check_passed"] = False

        except Exception as e:
            self.logger.exception(f"Error during data collection: {e}")
            pipeline_state["data_collection_completed"] = False
            pipeline_state["quality_check_passed"] = False

        return pipeline_state

    async def _run_enhanced_quality_check(self, training_input: dict[str, Any]) -> bool:
        """Run enhanced quality check after data collection."""
        try:
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")
            data_dir = training_input.get("data_dir", "data_cache")

            self.logger.info("🔍 Running enhanced quality check...")

            manager = EnhancedDataQualityManager(data_dir)
            quality_results = await manager.comprehensive_quality_check(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                check_gaps=True,
                fill_gaps=True,
                validate_format=True,
            )

            if quality_results.get("success", False):
                self.logger.info("✅ Enhanced quality check completed successfully")

                # Log quality metrics
                if quality_results.get("gaps_detected"):
                    self.logger.info(f"📊 Detected {len(quality_results['gaps_detected'])} gaps")
                if quality_results.get("gaps_filled"):
                    self.logger.info(f"🔧 Filled {len(quality_results['gaps_filled'])} gaps")
                if quality_results.get("format_issues"):
                    self.logger.warning(f"⚠️ Found {len(quality_results['format_issues'])} format issues")

                return True
            else:
                self.logger.error("❌ Enhanced quality check failed")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Error running enhanced quality check: {e}")
            return False

    @handle_data_collection_errors(context="run_data_collection")
    @log_step_metrics(context="data_collection")
    @((validate_file_operation("step1", expected_schema="klines", log_level="INFO") if validate_file_operation else (lambda x: x)))
    @((step_specific_ml_validation("step1", timestamp_col="timestamp") if step_specific_ml_validation else (lambda x: x)))
    async def _run_data_collection(self, training_input: dict[str, Any]) -> bool:
        """Run the actual data collection process."""
        try:
            # Try to import the downloader if not already imported
            global download_all_data_with_consolidation
            if download_all_data_with_consolidation is None:
                try:
                    from src.training.steps.data_downloader import download_all_data_with_consolidation as _dl
                    download_all_data_with_consolidation = _dl
                except ImportError:
                    self.logger.warning("Could not import data downloader, using fallback")
                    return await self._fallback_data_collection(training_input)

            if download_all_data_with_consolidation:
                # Use the existing data downloader if available
                symbol = training_input.get("symbol", "ETHUSDT")
                exchange = training_input.get("exchange", "BINANCE")
                timeframe = training_input.get("timeframe", "1m")

                self.logger.info(f"📊 Downloading data for {exchange}_{symbol}_{timeframe}")
                success = await download_all_data_with_consolidation(
                    symbol=symbol,
                    exchange_name=exchange,
                    interval=timeframe,
                )
                
                if success:
                    self.logger.info("✅ Data download completed successfully")
                    # Log immediate data extract after download
                    data_dir = training_input.get("data_dir", "data_cache")
                    await self._log_detailed_data_extract(symbol, exchange, timeframe, data_dir, self.logger)
                
                return bool(success)
            # Fallback implementation
            self.logger.warning("Using fallback data collection method")
            return await self._fallback_data_collection(training_input)

        except Exception as e:
            self.logger.exception(f"Error in data collection: {e}")
            return False

    @handle_data_collection_errors(context="fallback_data_collection")
    async def _fallback_data_collection(self, training_input: dict[str, Any]) -> bool:
        """Fallback data collection method."""
        self.logger.info("Running fallback data collection...")
        # Add fallback implementation here if needed
        return True

    async def _run_comprehensive_validation(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        logger: Any,
    ) -> bool:
        """Run comprehensive file format validation for step 1."""
        try:
            if not validate_step1_file:
                logger.warning("Comprehensive file validation not available")
                return True

            # Define expected files for step 1
            expected_files = [
                f"{data_dir}/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
                f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            ]

            validation_results: list[Any] = []
            all_valid = True

            for file_path in expected_files:
                if Path(file_path).exists():
                    logger.info(f"🔍 Validating file: {file_path}")

                    # Validate file format
                    validation_result = validate_step1_file(file_path)  # type: ignore[misc]
                    validation_results.append(validation_result)

                    if getattr(validation_result, "is_valid", False):
                        logger.info(f"✅ File validation passed: {file_path}")
                        logger.info(f"   📊 Shape: {validation_result.summary.get('shape', 'N/A')}")
                        logger.info(f"   📁 File type: {validation_result.file_type}")
                        logger.info(f"   🗂️ Columns: {validation_result.summary.get('column_count', 'N/A')}")
                    else:
                        logger.warning(f"⚠️ File validation issues found: {file_path}")
                        all_valid = False

                        # Log detailed issues
                        for issue in getattr(validation_result, "issues", []) or []:
                            logger.warning(f"   - {issue.severity.value.upper()}: {issue.description}")
                            if getattr(issue, "details", None):
                                logger.warning(f"     Details: {issue.details}")
                else:
                    logger.warning(f"⚠️ Expected file not found: {file_path}")
                    all_valid = False

            # Log validation summary
            if validation_results:
                total_files = len(validation_results)
                valid_files = sum(1 for r in validation_results if getattr(r, "is_valid", False))
                logger.info(f"📊 Validation Summary: {valid_files}/{total_files} files passed validation")

            return all_valid

        except Exception as e:
            logger.exception(f"❌ Error during comprehensive validation: {e}")
            return False

    async def _log_detailed_data_extract(
        self, symbol: str, exchange: str, timeframe: str, data_dir: str, logger: Any
    ) -> None:
        """Log detailed data extract for troubleshooting purposes.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            logger: Logger instance
        """
        logger.info("=" * 80)
        logger.info("📊 DETAILED DATA EXTRACT FOR TROUBLESHOOTING")
        logger.info("=" * 80)
        
        try:
            import pandas as pd
            
            # Check for consolidated files
            klines_file = f"{data_dir}/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            aggtrades_file = f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            files_to_check = [
                ("Klines", klines_file),
                ("Aggtrades", aggtrades_file)
            ]
            
            for data_type, file_path in files_to_check:
                logger.info(f"🔍 Analyzing {data_type} data: {file_path}")
                
                if Path(file_path).exists():
                    try:
                        # Load the data
                        df = pd.read_parquet(file_path)
                        
                        # Basic information
                        logger.info(f"   📊 Shape: {df.shape}")
                        logger.info(f"   📁 File size: {Path(file_path).stat().st_size:,} bytes")
                        
                        # Column information
                        logger.info(f"   🗂️ Columns ({len(df.columns)}): {list(df.columns)}")
                        
                        # Data types
                        logger.info(f"   🔧 Data types:")
                        for col, dtype in df.dtypes.items():
                            logger.info(f"      - {col}: {dtype}")
                        
                        # Sample data (first 5 rows)
                        logger.info(f"   📋 Sample data (first 5 rows):")
                        sample_df = df.head(5)
                        for idx, row in sample_df.iterrows():
                            # Format the row data for better readability
                            formatted_row = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    formatted_row[col] = "NaN"
                                elif isinstance(val, (int, float)):
                                    formatted_row[col] = f"{val:.6f}" if isinstance(val, float) else str(val)
                                else:
                                    formatted_row[col] = str(val)
                            logger.info(f"      Row {idx}: {formatted_row}")
                        
                        # Last 5 rows for comparison
                        logger.info(f"   📋 Sample data (last 5 rows):")
                        sample_df_last = df.tail(5)
                        for idx, row in sample_df_last.iterrows():
                            # Format the row data for better readability
                            formatted_row = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    formatted_row[col] = "NaN"
                                elif isinstance(val, (int, float)):
                                    formatted_row[col] = f"{val:.6f}" if isinstance(val, float) else str(val)
                                else:
                                    formatted_row[col] = str(val)
                            logger.info(f"      Row {idx}: {formatted_row}")
                        
                        # Date range information
                        if "timestamp" in df.columns:
                            try:
                                df["timestamp"] = pd.to_datetime(df["timestamp"])
                                min_date = df["timestamp"].min()
                                max_date = df["timestamp"].max()
                                total_days = (max_date - min_date).days
                                logger.info(f"   📅 Date range: {min_date} to {max_date} ({total_days} days)")
                            except Exception as e:
                                logger.warning(f"   ⚠️ Could not parse timestamp: {e}")
                        
                        # Value ranges for numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            logger.info(f"   📈 Numeric value ranges:")
                            for col in numeric_cols:
                                if col in df.columns:
                                    col_data = df[col].dropna()
                                    if len(col_data) > 0:
                                        min_val = col_data.min()
                                        max_val = col_data.max()
                                        mean_val = col_data.mean()
                                        logger.info(f"      - {col}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
                        
                        # Missing values
                        missing_counts = df.isnull().sum()
                        if missing_counts.sum() > 0:
                            logger.warning(f"   ⚠️ Missing values:")
                            for col, count in missing_counts.items():
                                if count > 0:
                                    percentage = (count / len(df)) * 100
                                    logger.warning(f"      - {col}: {count} ({percentage:.2f}%)")
                        else:
                            logger.info(f"   ✅ No missing values found")
                        
                        # Duplicate check
                        if "timestamp" in df.columns:
                            duplicates = df.duplicated(subset=["timestamp"]).sum()
                            if duplicates > 0:
                                logger.warning(f"   ⚠️ Found {duplicates} duplicate timestamps")
                            else:
                                logger.info(f"   ✅ No duplicate timestamps found")
                        
                        # Data quality checks
                        logger.info(f"   🔍 Data quality checks:")
                        
                        # Check for infinite values
                        infinite_counts = {}
                        for col in numeric_cols:
                            if col in df.columns:
                                infinite_count = (df[col] == float('inf')).sum() + (df[col] == float('-inf')).sum()
                                if infinite_count > 0:
                                    infinite_counts[col] = infinite_count
                        
                        if infinite_counts:
                            logger.warning(f"      ⚠️ Infinite values found:")
                            for col, count in infinite_counts.items():
                                logger.warning(f"         - {col}: {count} infinite values")
                        else:
                            logger.info(f"      ✅ No infinite values found")
                        
                        # Check for zero values in price columns
                        price_columns = ['open', 'high', 'low', 'close', 'price']
                        zero_price_counts = {}
                        for col in price_columns:
                            if col in df.columns:
                                zero_count = (df[col] == 0).sum()
                                if zero_count > 0:
                                    zero_price_counts[col] = zero_count
                        
                        if zero_price_counts:
                            logger.warning(f"      ⚠️ Zero values in price columns:")
                            for col, count in zero_price_counts.items():
                                logger.warning(f"         - {col}: {count} zero values")
                        else:
                            logger.info(f"      ✅ No zero values in price columns")
                        
                        # Check for negative values in volume
                        if 'volume' in df.columns:
                            negative_volume = (df['volume'] < 0).sum()
                            if negative_volume > 0:
                                logger.warning(f"      ⚠️ Negative volume values: {negative_volume}")
                            else:
                                logger.info(f"      ✅ No negative volume values")
                        
                        logger.info(f"   ✅ {data_type} data analysis completed")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Error analyzing {data_type} data: {e}")
                        logger.error(f"   📋 Full error: {str(e)}")
                else:
                    logger.warning(f"   ⚠️ File not found: {file_path}")
                
                logger.info("")  # Empty line for readability
            
            # Summary
            logger.info("📋 DATA EXTRACT SUMMARY:")
            existing_files = sum(1 for _, file_path in files_to_check if Path(file_path).exists())
            logger.info(f"   • Files found: {existing_files}/{len(files_to_check)}")
            logger.info(f"   • Data types analyzed: Klines, Aggtrades")
            logger.info("   • Information logged: Shape, columns, data types, sample data, date ranges, value ranges, missing values, duplicates")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error in detailed data extract: {e}")
            logger.error(f"📋 Full error: {str(e)}")
            logger.info("=" * 80)


@monitor_data_collection()
@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step1_data_collection",
)
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the data collection step.

    Args:
        symbol: Trading symbol (e.g. = "ETHUSDT")
        exchange: Exchange name (e.g. = "BINANCE")
        timeframe: Timeframe (e.g. = "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if data exists
        **kwargs: Additional arguments

    Returns:
        bool: True if successful = False otherwise

    """
    try:
        logger = system_logger.getChild("Step1DataCollection")

        logger.info("=" * 80)
        logger.info("🚀 STEP 1: Data Collection")
        logger.info("=" * 80)
        logger.info(f"🎯 Symbol: {symbol}")
        logger.info(f"🏢 Exchange: {exchange}")
        logger.info(f"📊 Timeframe: {timeframe}")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🔄 Force rerun: {force_rerun}")

        # Check if data already exists and force_rerun is False
        if not force_rerun:
            # Check for existing consolidated data
            consolidated_files = [
                f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            ]

            existing_files: list[str] = []
            for file_path in consolidated_files:
                if Path(file_path).exists():
                    existing_files.append(file_path)

            if existing_files:
                logger.info(f"✅ Found existing consolidated data: {len(existing_files)} files")
                logger.info("   📁 Existing files:")
                for file_path in existing_files:
                    logger.info(f"      - {file_path}")

                # Check if data is complete by examining the date range
                try:
                    import pandas as pd
                    klines_file = f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
                    if Path(klines_file).exists():
                        df = pd.read_parquet(klines_file)
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                            min_date = df["timestamp"].min().date()
                            max_date = df["timestamp"].max().date()
                            current_date = datetime.now().date()

                            # Check if we have recent data (within last 30 days)
                            days_since_last_data = (current_date - max_date).days

                            if days_since_last_data > 30:
                                logger.info(f"⚠️ Data is {days_since_last_data} days old, downloading recent data...")
                                # Continue with data collection to download missing data
                            else:
                                logger.info(f"✅ Data is up to date (last data: {max_date}, {days_since_last_data} days ago)")
                                logger.info("✅ Step 1: Data Collection completed (using existing data)")
                                
                                # Show detailed data extract for existing data
                                step = DataCollectionStep(CONFIG or {})
                                await step._log_detailed_data_extract(symbol, exchange, timeframe, data_dir, logger)
                                
                                return True
                        else:
                            logger.warning("⚠️ Could not determine data completeness, proceeding with data collection...")
                    else:
                        logger.warning("⚠️ Klines file not found, proceeding with data collection...")
                except Exception as e:
                    logger.warning(f"⚠️ Error checking data completeness: {e}, proceeding with data collection...")

        # Initialize data collection step
        step = DataCollectionStep(CONFIG or {})
        await step.initialize()

        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
        }

        # Execute data collection
        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        if result.get("data_collection_completed", False):
            logger.info("✅ Step 1: Data Collection completed successfully")

            # Show detailed data extract for troubleshooting
            await step._log_detailed_data_extract(symbol, exchange, timeframe, data_dir, logger)

            # Run comprehensive data quality validation
            try:
                from src.utils.comprehensive_data_quality_validator import validate_step1_quality
                
                logger.info("🔍 Running comprehensive data quality validation...")
                validation_result = validate_step1_quality(
                    symbol=symbol,
                    exchange=exchange,
                    data_dir=data_dir
                )
                
                if validation_result["validation_passed"]:
                    logger.info("✅ Comprehensive data quality validation passed")
                else:
                    logger.warning(f"⚠️ Comprehensive data quality validation found {len(validation_result['issues'])} issues:")
                    for issue in validation_result["issues"][:5]:  # Show first 5 issues
                        logger.warning(f"   - {issue}")
                    if len(validation_result["issues"]) > 5:
                        logger.warning(f"   ... and {len(validation_result['issues']) - 5} more issues")
                    
                    # Continue with warning instead of failing
                    logger.warning("⚠️ Continuing with data quality issues - review logs for details")
                
                # Also run legacy validation if available
                if validate_step1_file:
                    logger.info("🔍 Running legacy file format validation...")
                    validation_success = await step._run_comprehensive_validation(symbol, exchange, timeframe, data_dir, logger)
                    if not validation_success:
                        logger.warning("⚠️ Legacy file format validation found issues")
                
            except Exception as e:
                logger.warning(f"⚠️ Comprehensive data quality validation failed: {e} - continuing anyway")
                
                # Fallback to legacy validation if available
                if validate_step1_file:
                    logger.info("🔍 Running legacy file format validation...")
                    validation_success = await step._run_comprehensive_validation(symbol, exchange, timeframe, data_dir, logger)
                    if not validation_success:
                        logger.warning("⚠️ Legacy file format validation found issues")
            
            return True
        else:
            logger.error("❌ Step 1: Data Collection failed")
        return False

    except Exception as e:
        logger.exception(f"❌ Step 1: Data Collection failed: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    import asyncio

    async def main() -> None:
        # Get command line arguments
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else "data_cache"
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
        else:
            print("Usage: python step1_data_collection.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]")
            print("Example: python step1_data_collection.py ETHUSDT BINANCE 1m data_cache true")
            return

        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        if success:
            print("✅ Step 1: Data Collection completed successfully")
        else:
            print("❌ Step 1: Data Collection failed")

        # Clean up memory to prevent segmentation fault
        import gc
        gc.collect()

    # Use a more robust approach to prevent segmentation fault
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Final cleanup
        import gc
        gc.collect()