#!/usr/bin/env python3
"""Enhanced Data Quality Manager for Step1 and Step1_5.

This module provides comprehensive data quality management including:
- Data gap detection and filling
- Data quality validation and formatting
- Efficient processing with proper decorators
- Integration with step3/step4 data requirements
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    optimize_memory_usage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
)
from src.utils.logger import system_logger

logger = system_logger.getChild("EnhancedDataQualityManager")


class EnhancedDataQualityManager:
    """Comprehensive data quality manager with gap detection, filling, and validation."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

        # Initialize components
        self.gap_detector = None
        self.gap_filler = None
        self.validator = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all quality management components."""
        try:
            from .data_gap_detector import DataGapDetector
            self.gap_detector = DataGapDetector(str(self.data_cache_path))
        except ImportError as e:
            logger.warning(f"⚠️ Could not import DataGapDetector: {e}")

        try:
            from .comprehensive_gap_filler import ComprehensiveGapFiller
            self.gap_filler = ComprehensiveGapFiller(str(self.data_cache_path))
        except ImportError as e:
            logger.warning(f"⚠️ Could not import ComprehensiveGapFiller: {e}")

        try:
            from .aggtrades_validator import AggtradesValidator
            self.validator = AggtradesValidator(str(self.data_cache_path))
        except ImportError as e:
            logger.warning(f"⚠️ Could not import AggtradesValidator: {e}")

    @with_tracing_span("comprehensive_data_quality_check")
    @quality_gate(
        min_quality_score=0.6,
        max_correlation=0.95,
        required_grade="C"
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "issues": ["Quality check failed"]},
        context="enhanced_data_quality_manager.comprehensive_quality_check"
    )
    async def comprehensive_quality_check(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        check_gaps: bool = True,
        fill_gaps: bool = True,
        validate_format: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive data quality check with gap detection and filling.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            check_gaps: Whether to check for data gaps
            fill_gaps: Whether to automatically fill detected gaps
            validate_format: Whether to validate data format

        Returns:
            Dictionary with quality check results
        """
        logger.info(f"🔍 Starting comprehensive quality check for {exchange}_{symbol}_{timeframe}")

        results = {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "gaps_detected": [],
            "gaps_filled": [],
            "format_issues": [],
            "quality_metrics": {},
            "recommendations": []
        }

        try:
            # Step 1: Check for data gaps
            if check_gaps and self.gap_detector:
                gap_results = await self._check_data_gaps(symbol, exchange, timeframe)
                results["gaps_detected"] = gap_results.get("gaps", [])

                if gap_results.get("gaps"):
                    logger.warning(f"⚠️ Found {len(gap_results['gaps'])} data gaps")
                    results["recommendations"].append("Data gaps detected - consider filling them")

                    # Step 2: Fill gaps if requested
                    if fill_gaps and self.gap_filler:
                        fill_results = await self._fill_data_gaps(symbol, exchange, timeframe, gap_results["gaps"])
                        results["gaps_filled"] = fill_results.get("filled_gaps", [])

                        if fill_results.get("success"):
                            logger.info(f"✅ Successfully filled {len(fill_results['filled_gaps'])} gaps")
                        else:
                            logger.error("❌ Failed to fill some data gaps")
                            results["success"] = False

            # Step 3: Validate data format
            if validate_format and self.validator:
                format_results = await self._validate_data_format(symbol, exchange, timeframe)
                results["format_issues"] = format_results.get("issues", [])
                results["quality_metrics"] = format_results.get("metrics", {})

                if format_results.get("issues"):
                    logger.warning(f"⚠️ Found {len(format_results['issues'])} format issues")
                    results["recommendations"].append("Data format issues detected - consider fixing them")

            # Step 4: Check data completeness for step3/step4 requirements
            completeness_results = await self._check_step3_step4_completeness(symbol, exchange, timeframe)
            results["step3_step4_ready"] = completeness_results.get("ready", False)
            results["missing_for_steps"] = completeness_results.get("missing", [])

            if not completeness_results.get("ready"):
                results["recommendations"].append("Data not ready for step3/step4 - additional data needed")

            logger.info(f"✅ Comprehensive quality check completed for {exchange}_{symbol}_{timeframe}")
            return results

        except Exception as e:
            logger.exception(f"❌ Comprehensive quality check failed: {e}")
            results["success"] = False
            results["issues"].append(f"Quality check failed: {str(e)}")
            return results

    @with_tracing_span("check_data_gaps")
    @memory_efficient
    async def _check_data_gaps(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Check for data gaps using the gap detector."""
        if not self.gap_detector:
            return {"gaps": [], "error": "Gap detector not available"}

        try:
            # Check for missing data periods
            missing_data = self.gap_detector.detect_missing_data(symbol, exchange)

            gaps = []

            # Process missing aggtrades
            for day in missing_data.get("missing_aggtrades_days", []):
                gaps.append({
                    "type": "aggtrades",
                    "date": day,
                    "severity": "high",
                    "description": f"Missing aggtrades data for {day}"
                })

            # Process missing klines
            for month in missing_data.get("missing_klines_months", []):
                gaps.append({
                    "type": "klines",
                    "date": month,
                    "severity": "high",
                    "description": f"Missing klines data for {month}"
                })

            # Process missing futures
            for month in missing_data.get("missing_futures_months", []):
                gaps.append({
                    "type": "futures",
                    "date": month,
                    "severity": "medium",
                    "description": f"Missing futures data for {month}"
                })

            return {"gaps": gaps, "total_gaps": len(gaps)}

        except Exception as e:
            logger.exception(f"❌ Error checking data gaps: {e}")
            return {"gaps": [], "error": str(e)}

    @with_tracing_span("fill_data_gaps")
    @resource_monitor
    async def _fill_data_gaps(self, symbol: str, exchange: str, timeframe: str, gaps: List[Dict]) -> Dict[str, Any]:
        """Fill detected data gaps using the gap filler."""
        if not self.gap_filler:
            return {"filled_gaps": [], "error": "Gap filler not available"}

        try:
            filled_gaps = []

            for gap in gaps:
                try:
                    if gap["type"] == "aggtrades":
                        # Fill aggtrades gap
                        success = await self.gap_filler.fill_aggtrades_gap(
                            symbol, exchange, gap["date"]
                        )
                        if success:
                            filled_gaps.append(gap)

                    elif gap["type"] == "klines":
                        # Fill klines gap
                        success = await self.gap_filler.fill_klines_gap(
                            symbol, exchange, timeframe, gap["date"]
                        )
                        if success:
                            filled_gaps.append(gap)

                    elif gap["type"] == "futures":
                        # Fill futures gap
                        success = await self.gap_filler.fill_futures_gap(
                            symbol, exchange, gap["date"]
                        )
                        if success:
                            filled_gaps.append(gap)

                except Exception as e:
                    logger.warning(f"⚠️ Failed to fill gap {gap}: {e}")

            return {
                "filled_gaps": filled_gaps,
                "total_filled": len(filled_gaps),
                "success": len(filled_gaps) == len(gaps)
            }

        except Exception as e:
            logger.exception(f"❌ Error filling data gaps: {e}")
            return {"filled_gaps": [], "error": str(e)}

    @with_tracing_span("validate_data_format")
    @validate_data_structure
    async def _validate_data_format(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate data format using the validator."""
        if not self.validator:
            return {"issues": [], "metrics": {}, "error": "Validator not available"}

        try:
            issues = []
            metrics = {}

            # Validate aggtrades files
            aggtrades_files = self.validator.get_aggtrades_files(symbol, exchange)
            for file_path in aggtrades_files:
                validation_result = self.validator.validate_file_format(file_path)
                if not validation_result.get("valid", False):
                    issues.extend(validation_result.get("issues", []))

                # Collect metrics
                metrics[f"aggtrades_{file_path.name}"] = {
                    "file_size": validation_result.get("file_size", 0),
                    "row_count": validation_result.get("row_count", 0),
                    "valid": validation_result.get("valid", False)
                }

            # Validate klines files
            klines_pattern = f"klines_{exchange}_{symbol}_{timeframe}_*.parquet"
            klines_files = list(self.data_cache_path.glob(klines_pattern))

            for file_path in klines_files:
                try:
                    df = pd.read_parquet(file_path)
                    metrics[f"klines_{file_path.name}"] = {
                        "file_size": file_path.stat().st_size,
                        "row_count": len(df),
                        "valid": True,
                        "columns": list(df.columns),
                        "date_range": {
                            "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
                            "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None
                        }
                    }
                except Exception as e:
                    issues.append(f"Invalid klines file {file_path.name}: {e}")
                    metrics[f"klines_{file_path.name}"] = {
                        "valid": False,
                        "error": str(e)
                    }

            return {"issues": issues, "metrics": metrics}

        except Exception as e:
            logger.exception(f"❌ Error validating data format: {e}")
            return {"issues": [f"Validation failed: {e}"], "metrics": {}}

    @with_tracing_span("check_step3_step4_completeness")
    @comprehensive_data_validation
    async def _check_step3_step4_completeness(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Check if data is complete for step3 and step4 requirements."""
        try:
            missing = []
            ready = True

            # Check for unified data (required by step01_5)
            unified_path = self.data_cache_path / "unified" / exchange.lower() / symbol / timeframe
            if not unified_path.exists():
                missing.append("Unified data directory not found")
                ready = False

            # Check for minimum data requirements for HMM (step3)
            klines_file = self.data_cache_path / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            if klines_file.exists():
                try:
                    df = pd.read_parquet(klines_file)
                    if len(df) < 10000:  # Minimum rows for HMM
                        missing.append("Insufficient klines data for HMM analysis")
                        ready = False

                    # Check for required columns
                    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        missing.append(f"Missing required columns: {missing_columns}")
                        ready = False

                except Exception as e:
                    missing.append(f"Error reading klines file: {e}")
                    ready = False
            else:
                missing.append("Klines consolidated file not found")
                ready = False

            # Check for aggtrades data (required for step4 labeling)
            aggtrades_file = self.data_cache_path / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            if not aggtrades_file.exists():
                missing.append("Aggtrades consolidated file not found")
                ready = False

            return {
                "ready": ready,
                "missing": missing,
                "unified_data_exists": unified_path.exists(),
                "klines_available": klines_file.exists(),
                "aggtrades_available": aggtrades_file.exists()
            }

        except Exception as e:
            logger.exception(f"❌ Error checking step3/step4 completeness: {e}")
            return {
                "ready": False,
                "missing": [f"Completeness check failed: {e}"],
                "unified_data_exists": False,
                "klines_available": False,
                "aggtrades_available": False
            }

    @with_tracing_span("get_data_for_step3_step4")
    @secure_data_processing
    async def get_data_for_step3_step4(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Get data ready for step3 and step4, ensuring all gaps are filled and quality is validated."""
        logger.info(f"📊 Preparing data for step3/step4: {exchange}_{symbol}_{timeframe}")

        try:
            # First, run comprehensive quality check
            quality_results = await self.comprehensive_quality_check(
                symbol, exchange, timeframe,
                check_gaps=True, fill_gaps=True, validate_format=True
            )

            if not quality_results.get("success", False):
                logger.error("❌ Data quality check failed")
                return {
                    "success": False,
                    "error": "Data quality check failed",
                    "issues": quality_results.get("issues", [])
                }

            # Check if data is ready for step3/step4
            completeness_results = await self._check_step3_step4_completeness(symbol, exchange, timeframe)

            if not completeness_results.get("ready", False):
                logger.warning("⚠️ Data not ready for step3/step4, attempting to fix...")

                # Try to use step1 and step01_5 components to get missing data
                fix_results = await self._fix_missing_data_for_steps(symbol, exchange, timeframe)

                if not fix_results.get("success", False):
                    return {
                        "success": False,
                        "error": "Failed to prepare data for step3/step4",
                        "missing": completeness_results.get("missing", []),
                        "fix_attempts": fix_results
                    }

            # Return data paths and metadata
            return {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "unified_data_path": str(self.data_cache_path / "unified" / exchange.lower() / symbol / timeframe),
                "klines_path": str(self.data_cache_path / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"),
                "aggtrades_path": str(self.data_cache_path / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"),
                "quality_metrics": quality_results.get("quality_metrics", {}),
                "ready_for_steps": True
            }

        except Exception as e:
            logger.exception(f"❌ Error preparing data for step3/step4: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @with_tracing_span("fix_missing_data_for_steps")
    async def _fix_missing_data_for_steps(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Use step1 and step01_5 components to fix missing data for step3/step4."""
        try:
            logger.info("🔄 Attempting to fix missing data using step1/step01_5 components...")

            # Try to run step1 data collection if needed
            try:
                from ..step1_data_collection import run_step as run_step1
                step1_success = await run_step1(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=True
                )

                if step1_success:
                    logger.info("✅ Step1 data collection completed successfully")
                else:
                    logger.warning("⚠️ Step1 data collection failed")

            except Exception as e:
                logger.warning(f"⚠️ Could not run step1: {e}")
                step1_success = False

            # Try to run step01_5 data conversion if needed
            try:
                from ..step01_5_data_converter import run_step as run_step01_5
                step01_5_success = await run_step01_5(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=True
                )

                if step01_5_success:
                    logger.info("✅ Step1_5 data conversion completed successfully")
                else:
                    logger.warning("⚠️ Step1_5 data conversion failed")

            except Exception as e:
                logger.warning(f"⚠️ Could not run step01_5: {e}")
                step01_5_success = False

            # Check if data is now ready
            completeness_results = await self._check_step3_step4_completeness(symbol, exchange, timeframe)

            return {
                "success": completeness_results.get("ready", False),
                "step1_success": step1_success,
                "step01_5_success": step01_5_success,
                "still_missing": completeness_results.get("missing", [])
            }

        except Exception as e:
            logger.exception(f"❌ Error fixing missing data: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Convenience function for easy integration
@with_tracing_span("ensure_data_quality")
async def ensure_data_quality(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_cache_path: str = "data_cache"
) -> Dict[str, Any]:
    """Convenience function to ensure data quality for a given symbol/exchange/timeframe."""
    manager = EnhancedDataQualityManager(data_cache_path)
    return await manager.comprehensive_quality_check(symbol, exchange, timeframe)


# Convenience function for step3/step4 integration
@with_tracing_span("prepare_data_for_steps")
async def prepare_data_for_steps(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_cache_path: str = "data_cache"
) -> Dict[str, Any]:
    """Convenience function to prepare data for step3/step4."""
    manager = EnhancedDataQualityManager(data_cache_path)
    return await manager.get_data_for_step3_step4(symbol, exchange, timeframe)