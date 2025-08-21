"""Step1 Orchestrator - Coordinates data collection and validation processes.

Coordinates data collection processes for step1. This orchestrator focuses on:
1. Detecting missing data gaps (aggtrades, klines, futures)
2. Validating data quality and format
3. Preparing data for step1_5_data_converter.py processing

Note: Data conversion and formatting is handled by step1_5_data_converter.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.centralized_decorators import (
    handle_errors,
    with_tracing_span,
)
from src.utils.logger import system_logger

from .aggtrades_validator import AggtradesValidator
from .comprehensive_gap_filler import ComprehensiveGapFiller
from .data_gap_detector import DataGapDetector
from .data_resampler import DataPreparation
from .missing_data_downloader_and_gap_filler import MissingDataDownloaderAndGapFiller

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("Step1Orchestrator")


class Step1Orchestrator:
    """Orchestrates step1 data collection processes with proper decorators and security."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

        # Initialize components
        self.gap_detector = DataGapDetector(data_cache_path)
        self.aggtrades_validator = AggtradesValidator(data_cache_path)
        self.data_preparation = DataPreparation(data_cache_path)
        self.data_downloader = MissingDataDownloaderAndGapFiller(data_cache_path)
        self.comprehensive_gap_filler = ComprehensiveGapFiller(data_cache_path)

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
            "success": False,
            "errors": ["Step1 orchestration failed"],
            "warnings": [],
            "step1_5_ready": False,
        },
        context="step1_orchestrator.run_complete_step1",
    )
    async def run_complete_step1(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        auto_fix: bool = True,
    ) -> dict:
        """Run complete step1 data collection process including:
        1. Detect missing data gaps (aggtrades, klines, futures)
        2. Validate data quality and format
        3. Prepare data for step1_5_data_converter.py processing.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date for analysis
            end_date: End date for analysis
            auto_fix: Whether to automatically fix issues

        Returns:
            Dictionary with step1 collection results

        """
        logger.info(f"🚀 STARTING COMPLETE STEP1 PROCESS FOR {exchange}_{symbol}")
        logger.info("=" * 80)

        results = {
            "symbol": symbol,
            "exchange": exchange,
            "start_date": start_date,
            "end_date": end_date,
            "success": True,
            "errors": [],
            "warnings": [],
            "step1_5_ready": False,
        }

        # Step 1.1: Comprehensive Gap Detection and Filling
        logger.info("📊 STEP 1.1: COMPREHENSIVE GAP DETECTION AND FILLING")
        logger.info("-" * 60)

        try:
            gap_filling_results = await self.comprehensive_gap_filler.process_all_data_types(symbol, exchange)

            if gap_filling_results:
                results["gap_filling"] = gap_filling_results
                logger.info(f"✅ Gap filling completed: {gap_filling_results.get('gaps_filled', 0)} gaps filled")
            else:
                logger.warning("⚠️ Gap filling returned no results")
                results["warnings"].append("Gap filling returned no results")

        except Exception as e:
            logger.exception(f"❌ Gap filling failed: {e}")
            results["errors"].append(f"Gap filling failed: {e}")
            results["success"] = False

        try:
            # Step 1: Detect missing data gaps
            logger.info("📊 STEP 1.1: DETECTING MISSING DATA GAPS")
            logger.info("-" * 60)

            missing_data = self.gap_detector.detect_missing_data(
                symbol, exchange, start_date, end_date,
            )
            results["missing_data"] = missing_data

            # Check for critical missing data
            total_missing = (
                len(missing_data["missing_aggtrades_days"])
                + len(missing_data["missing_klines_months"])
                + len(missing_data["missing_futures_months"])
            )

            if total_missing > 0:
                results["warnings"].append(
                    f"Found {total_missing} missing data periods",
                )
                logger.warning(f"⚠️ Found {total_missing} missing data periods")

            # Step 2: Download missing data and fill gaps
            if total_missing > 0 or len(missing_data.get("aggtrades_gaps", [])) > 0:
                logger.info("📊 STEP 1.2: DOWNLOADING MISSING DATA AND FILLING GAPS")
                logger.info("-" * 60)

                # Run async download process
                download_results = asyncio.run(
                    self.data_downloader.download_all_missing_data(
                        symbol, exchange, end_date,
                    ),
                )
                results["download_results"] = download_results

                if download_results["success"]:
                    logger.info("✅ Missing data download completed successfully")
                else:
                    logger.warning("⚠️ Missing data download encountered issues")
                    results["warnings"].append("Some data downloads failed")

            # Step 3: Detect aggtrades gaps (after potential downloads)
            logger.info("📊 STEP 1.3: DETECTING AGGTRADES GAPS")
            logger.info("-" * 60)

            aggtrades_gaps = self.gap_detector.detect_aggtrades_gaps(symbol, exchange)
            results["aggtrades_gaps"] = aggtrades_gaps

            if aggtrades_gaps:
                results["warnings"].append(
                    f"Found {len(aggtrades_gaps)} gaps in aggtrades data",
                )
                logger.warning(f"⚠️ Found {len(aggtrades_gaps)} gaps in aggtrades data")

            # Step 4: Aggtrades validation and fixing
            logger.info("📊 STEP 1.4: AGGTRADES VALIDATION AND FIXING")
            logger.info("-" * 60)

            aggtrades_validation = self.aggtrades_validator.validate_all_aggtrades(
                symbol, exchange, auto_fix=auto_fix,
            )
            results["aggtrades_validation"] = aggtrades_validation

            if aggtrades_validation["invalid_files"] > 0:
                if auto_fix:
                    logger.info(
                        f"🔧 Auto-fixed {aggtrades_validation['fixed_files']} aggtrades files",
                    )
                else:
                    results["warnings"].append(
                        f"{aggtrades_validation['invalid_files']} aggtrades files need fixing",
                    )

            # Step 5: Convert to parquet if needed
            logger.info("📊 STEP 1.5: CONVERTING TO PARQUET FORMAT")
            logger.info("-" * 60)

            conversion_results = self.aggtrades_validator.convert_to_parquet(
                symbol, exchange,
            )
            results["parquet_conversion"] = conversion_results

            if conversion_results["converted_files"] > 0:
                logger.info(
                    f"✅ Converted {conversion_results['converted_files']} files to parquet",
                )

            # Step 5.5: Create 1m consolidated data
            logger.info("📊 STEP 1.5.5: CREATING 1M CONSOLIDATED DATA")
            logger.info("-" * 60)

            consolidation_results = self.data_preparation.create_1m_consolidated_data(
                symbol, exchange,
            )
            results["1m_consolidation"] = consolidation_results

            if consolidation_results["success"]:
                logger.info("✅ 1m consolidated data created successfully")
            else:
                logger.warning("⚠️ 1m consolidation encountered issues")
                results["warnings"].append("1m consolidation incomplete")

            # Step 6: Resample data to multiple timeframes
            logger.info("📊 STEP 1.6: RESAMPLING DATA TO MULTIPLE TIMEFRAMES")
            logger.info("-" * 60)

            resampling_results = self.data_preparation.resample_all_timeframes(
                symbol,
                exchange,
                timeframes=["5m", "15m", "30m"],
                start_date=start_date,
                end_date=end_date,
                create_partitions=True,
            )
            results["resampling"] = resampling_results

            if resampling_results["success"]:
                logger.info(
                    f"✅ Resampling completed: {len(resampling_results['resampled_files'])} timeframes",
                )
            else:
                logger.warning("⚠️ Resampling encountered issues")
                results["warnings"].append("Resampling incomplete")

            # Step 6.5: Prepare data for step1_5 processing
            logger.info("📊 STEP 1.6.5: PREPARING DATA FOR STEP1_5 PROCESSING")
            logger.info("-" * 60)

            preparation_results = self.data_preparation.prepare_for_step1_5(
                symbol, exchange,
            )
            results["data_preparation"] = preparation_results

            if preparation_results["ready"]:
                logger.info("✅ Data preparation completed successfully")
            else:
                logger.warning("⚠️ Data preparation encountered issues")
                results["warnings"].append("Data preparation incomplete")

            # Step 7: Validate step1_5 readiness
            logger.info("📊 STEP 1.7: VALIDATING STEP1_5 READINESS")
            logger.info("-" * 60)

            step1_5_readiness = self.validate_step1_5_readiness(symbol, exchange)
            results["step1_5_readiness"] = step1_5_readiness
            results["step1_5_ready"] = step1_5_readiness["ready"]

            if not step1_5_readiness["ready"]:
                results["warnings"].append("Step1_5 data preparation incomplete")
                logger.warning("⚠️ Data not fully ready for step1_5 processing")

            # Step 8: Generate comprehensive report
            logger.info("📊 STEP 1.8: GENERATING COMPREHENSIVE REPORT")
            logger.info("-" * 60)

            report = self.generate_comprehensive_report(symbol, exchange, results)
            results["report"] = report

            logger.info("=" * 80)
            if results["success"]:
                logger.info("🎉 STEP1 PROCESS COMPLETED SUCCESSFULLY!")
            else:
                logger.error("❌ STEP1 PROCESS COMPLETED WITH ERRORS!")

            return results

        except Exception as e:
            logger.exception(f"❌ Error in step1 process: {e}")
            results["success"] = False
            results["errors"].append(str(e))
            return results

    @with_tracing_span("validate_step1_5_readiness")
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
            "ready": False,
            "issues": ["Step1_5 readiness validation failed"],
            "required_files": [],
            "missing_files": [],
        },
        context="step1_orchestrator.validate_step1_5_readiness",
    )
    def validate_step1_5_readiness(self, symbol: str, exchange: str) -> dict:
        """Validate that the data is ready for step1_5_data_converter.py processing.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with readiness results

        """
        logger.info(f"🔍 Validating step1_5 compatibility for {exchange}_{symbol}")

        readiness_result = {
            "ready": True,
            "issues": [],
            "required_files": [],
            "missing_files": [],
        }

        # Check for required aggtrades files
        aggtrades_files = self.aggtrades_validator.get_aggtrades_files(symbol, exchange)
        if not aggtrades_files:
            readiness_result["ready"] = False
            readiness_result["issues"].append("No aggtrades files found")
            readiness_result["missing_files"].append("aggtrades files")
        else:
            readiness_result["required_files"].extend([f.name for f in aggtrades_files])

        # Check for required klines files
        klines_files = self.data_preparation.get_klines_files(symbol, exchange)
        if not klines_files:
            readiness_result["ready"] = False
            readiness_result["issues"].append("No klines files found")
            readiness_result["missing_files"].append("klines files")
        else:
            readiness_result["required_files"].extend([f.name for f in klines_files])

        # Check for basic data quality (step1_5 will handle resampling)
        # We only need to ensure raw data is available and properly formatted
        for file_path in aggtrades_files:
            validation_result = self.aggtrades_validator.validate_file_format(file_path)
            if not validation_result["valid"]:
                readiness_result["ready"] = False
                readiness_result["issues"].append(f"Invalid format: {file_path.name}")

        # Check for 1m consolidated data (should be created by step1)
        data_cache_path = Path("data_cache")
        consolidated_1m_path = (
            data_cache_path / f"klines_{exchange}_{symbol}_1m_consolidated.parquet"
        )

        if not consolidated_1m_path.exists():
            readiness_result["ready"] = False
            readiness_result["issues"].append("1m consolidated data not found")
            readiness_result["missing_files"].append("1m consolidated data")
        else:
            readiness_result["required_files"].append(consolidated_1m_path.name)

        if readiness_result["ready"]:
            logger.info("✅ Step1_5 readiness check passed")
        else:
            logger.warning("⚠️ Step1_5 readiness check found issues")
            for issue in readiness_result["issues"]:
                logger.warning(f"  - {issue}")

        return readiness_result

    @with_tracing_span("generate_comprehensive_report")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, AttributeError),
        default_return="❌ ERROR: Failed to generate comprehensive report",
        context="step1_orchestrator.generate_comprehensive_report",
    )
    def generate_comprehensive_report(
        self, symbol: str, exchange: str, results: dict,
    ) -> str:
        """Generate a comprehensive report of the step1 process.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            results: Step1 process results

        Returns:
            Comprehensive report string

        """
        report = f"""
🎯 COMPREHENSIVE STEP1 REPORT FOR {exchange}_{symbol}
{'='*80}

📊 PROCESS SUMMARY:
• Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}
• Symbol: {symbol}
• Exchange: {exchange}

📈 MISSING DATA ANALYSIS:
• Missing Aggtrades Days: {len(results['missing_data']['missing_aggtrades_days'])}
• Missing Klines Months: {len(results['missing_data']['missing_klines_months'])}
• Missing Futures Months: {len(results['missing_data']['missing_futures_months'])}

📥 DOWNLOAD RESULTS:
"""

        # Add download results if available
        if results.get("download_results"):
            download_data = results["download_results"]
            if "aggtrades" in download_data.get("download_results", {}):
                aggtrades = download_data["download_results"]["aggtrades"]
                report += f"""
• Aggtrades Downloads:
  - Downloaded Days: {aggtrades['downloaded_days']}
  - Failed Days: {aggtrades['failed_days']}
  - Total Rows: {aggtrades['total_rows']}
"""

            if "klines" in download_data.get("download_results", {}):
                klines = download_data["download_results"]["klines"]
                report += f"""
• Klines Downloads:
  - Downloaded Months: {klines['downloaded_months']}
  - Failed Months: {klines['failed_months']}
  - Total Rows: {klines['total_rows']}
"""

            if "futures" in download_data.get("download_results", {}):
                futures = download_data["download_results"]["futures"]
                report += f"""
• Futures Downloads:
  - Downloaded Months: {futures['downloaded_months']}
  - Failed Months: {futures['failed_months']}
  - Total Rows: {futures['total_rows']}
"""

            if "gap_filling_results" in download_data:
                gaps = download_data["gap_filling_results"]
                report += f"""
• Gap Filling:
  - Filled Gaps: {gaps['filled_gaps']}
  - Failed Gaps: {gaps['failed_gaps']}
  - Total Rows Added: {gaps['total_rows_added']}
"""

        report += f"""
⚠️ DATA GAPS:
• Aggtrades Gaps > 10s: {len(results['aggtrades_gaps'])}

🔧 AGGTRADES VALIDATION:
• Total Files: {results['aggtrades_validation']['total_files']}
• Valid Files: {results['aggtrades_validation']['valid_files']}
• Invalid Files: {results['aggtrades_validation']['invalid_files']}
• Fixed Files: {results['aggtrades_validation']['fixed_files']}

🔄 RESAMPLING RESULTS:
• Source Rows: {results.get('resampling', {}).get('source_rows', 0)}
• Success: {results.get('resampling', {}).get('success', False)}
• Resampled Files: {len(results.get('resampling', {}).get('resampled_files', {}))}
• Partitioned Datasets: {len(results.get('resampling', {}).get('partitioned_datasets', {}))}

🔍 STEP1_5 COMPATIBILITY:
• Compatible: {'✅ YES' if results['step1_5_ready'] else '❌ NO'}
• Required Files: {len(results.get('step1_5_readiness', {}).get('required_files', []))}
• Missing Files: {len(results.get('step1_5_readiness', {}).get('missing_files', []))}

"""

        # Add errors and warnings
        if results["errors"]:
            report += f"""
❌ ERRORS:
{chr(10).join(f'• {error}' for error in results['errors'])}
"""

        if results["warnings"]:
            report += f"""
⚠️ WARNINGS:
{chr(10).join(f'• {warning}' for warning in results['warnings'])}
"""

        # Add detailed missing data
        if results["missing_data"]["missing_aggtrades_days"]:
            report += f"""
📅 MISSING AGGTRADES DAYS (first 10):
{chr(10).join(f'• {date}' for date in results['missing_data']['missing_aggtrades_days'][:10])}
{'  ...' if len(results['missing_data']['missing_aggtrades_days']) > 10 else ''}
"""

        # Add data gaps
        if results["aggtrades_gaps"]:
            report += f"""
⚠️ DATA GAPS (first 5):
{chr(10).join(f'• {gap["file"]}: {gap["gap_start"]} to {gap["gap_end"]} ({gap["gap_duration_seconds"]:.1f}s)' for gap in results['aggtrades_gaps'][:5])}
{'  ...' if len(results['aggtrades_gaps']) > 5 else ''}
"""

        report += f"""
{'='*80}
"""

        return report

    @with_tracing_span("quick_health_check")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError),
        default_return={"healthy": False, "issues": ["Health check failed"], "recommendations": ["Check system status"]},
        context="step1_orchestrator.quick_health_check",
    )
    def quick_health_check(self, symbol: str, exchange: str) -> dict:
        """Perform a quick health check of the data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with health check results

        """
        logger.info(f"🔍 Performing quick health check for {exchange}_{symbol}")

        health_result = {"healthy": True, "issues": [], "recommendations": []}

        # Check for basic data availability
        aggtrades_files = self.aggtrades_validator.get_aggtrades_files(symbol, exchange)
        klines_files = self.data_preparation.get_klines_files(symbol, exchange)

        if not aggtrades_files:
            health_result["healthy"] = False
            health_result["issues"].append("No aggtrades files found")
            health_result["recommendations"].append("Download missing aggtrades data")

        if not klines_files:
            health_result["healthy"] = False
            health_result["issues"].append("No klines files found")
            health_result["recommendations"].append("Download missing klines data")

        # Check for resampled data
        for timeframe in ["5m", "15m", "30m"]:
            output_dir = self.data_cache_path / "resampled" / exchange / symbol
            filename = f"klines_{exchange}_{symbol}_{timeframe}_resampled.parquet"
            file_path = output_dir / filename

            if not file_path.exists():
                health_result["issues"].append(f"Missing resampled {timeframe} data")
                health_result["recommendations"].append(
                    f"Run resampling for {timeframe} timeframe",
                )

        if health_result["healthy"]:
            logger.info("✅ Health check passed")
        else:
            logger.warning("⚠️ Health check found issues")
            for issue in health_result["issues"]:
                logger.warning(f"  - {issue}")

        return health_result

    def get_step1_status(self, symbol: str, exchange: str) -> dict:
        """Get current status of step1 data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with step1 status

        """
        status = {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(),
            "data_available": {},
            "missing_data": {},
            "resampled_data": {},
            "overall_status": "unknown",
        }

        # Check aggtrades data
        aggtrades_files = self.aggtrades_validator.get_aggtrades_files(symbol, exchange)
        status["data_available"]["aggtrades"] = len(aggtrades_files)

        # Check klines data
        klines_files = self.data_preparation.get_klines_files(symbol, exchange)
        status["data_available"]["klines"] = len(klines_files)

        # Check resampled data
        for timeframe in ["5m", "15m", "30m"]:
            output_dir = self.data_cache_path / "resampled" / exchange / symbol
            filename = f"klines_{exchange}_{symbol}_{timeframe}_resampled.parquet"
            file_path = output_dir / filename

            status["resampled_data"][timeframe] = file_path.exists()

        # Determine overall status
        if (
            status["data_available"]["aggtrades"] > 0
            and status["data_available"]["klines"] > 0
            and all(status["resampled_data"].values())
        ):
            status["overall_status"] = "complete"
        elif (
            status["data_available"]["aggtrades"] > 0
            or status["data_available"]["klines"] > 0
        ):
            status["overall_status"] = "partial"
        else:
            status["overall_status"] = "missing"

        return status
