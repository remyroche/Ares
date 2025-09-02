#!/usr/bin/env python3
"""
Fix Aggtrades Gaps Script

This script specifically targets files with the most gaps and attempts to fill them
by downloading missing data from the exchange.
"""

from pathlib import Path
from src.utils.logger import system_logger
from typing import Dict, List
import asyncio
import sys

from src.training.steps.step1.data_gap_detector import DataGapDetector
from src.training.steps.step1.missing_data_downloader_and_gap_filler import MissingDataDownloaderAndGapFiller

# Add project root to path
project_root=Path(__file__).parent
sys.path.insert(0, str(project_root))

logger=system_logger.getChild("FixAggtradesGaps")


async def fix_specific_files_gaps(
    symbol: str="ETHUSDT", exchange: str="BINANCE", target_files: List[str] = None
) -> Dict:
    """
    Fix gaps in specific aggtrades files

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        target_files: List of specific files to fix (optional)

    Returns:
        Dictionary with results
    """
    logger.info("🚀 STARTING TARGETED GAP FIXING")
    logger.info("=" * 80)

    # Initialize components
    gap_detector=DataGapDetector()
    gap_filler=MissingDataDownloaderAndGapFiller()

    results={
        "files_processed": 0,
        "gaps_before": 0,
        "gaps_after": 0,
        "gaps_fixed": 0,
        "failed_fixes": 0,
        "errors": [],
    }

    try:
        # Step 1: Detect all gaps
        logger.info("📊 STEP 1: DETECTING CURRENT GAPS")
        logger.info("-" * 60)

        all_gaps=gap_detector.detect_aggtrades_gaps(symbol, exchange)
        results["gaps_before"] = len(all_gaps)

        logger.info(f"📊 Found {len(all_gaps)} total gaps across all files")

        # Filter gaps for target files if specified
        if target_files:
            target_gaps=[gap for gap in all_gaps if gap["file"] in target_files]
            logger.info(
                f"🎯 Filtering to {len(target_gaps)} gaps in target files: {target_files}"
            )
        else:
            target_gaps=all_gaps

        if not target_gaps:
            logger.info("✅ No gaps found in target files")
            return results

        # Group gaps by file
        gaps_by_file={}
        for gap in target_gaps:
            file_name = gap["file"]
            if file_name not in gaps_by_file:
                gaps_by_file[file_name] = []
            gaps_by_file[file_name].append(gap)

        # Step 2: Fix gaps file by file
        logger.info("🔧 STEP 2: FIXING GAPS")
        logger.info("-" * 60)

        for file_name, file_gaps in gaps_by_file.items():
            logger.info(f"🔧 Processing {file_name}: {len(file_gaps)} gaps")
            results["files_processed"] += 1

            try:
                # Attempt to fill gaps for this file
                fill_results=await gap_filler.fill_aggtrades_gaps(
                    symbol, exchange, file_gaps
                )

                results["gaps_fixed"] += fill_results["filled_gaps"]
                results["failed_fixes"] += fill_results["failed_gaps"]

                if fill_results["errors"]:
                    results["errors"].extend(fill_results["errors"])

                logger.info(
                    f"✅ {file_name}: {fill_results['filled_gaps']} gaps filled, {fill_results['failed_gaps']} failed"
                )

            except Exception as e:
                logger.error(f"❌ Error processing {file_name}: {e}")
                results["errors"].append(f"{file_name}: {e}")
                results["failed_fixes"] += len(file_gaps)

        # Step 3: Re-detect gaps to verify improvements
        logger.info("📊 STEP 3: VERIFYING RESULTS")
        logger.info("-" * 60)

        updated_gaps=gap_detector.detect_aggtrades_gaps(symbol, exchange)
        results["gaps_after"] = len(updated_gaps)

        # Calculate improvement
        gaps_eliminated=results["gaps_before"] - results["gaps_after"]
        improvement_rate = (
            (gaps_eliminated / results["gaps_before"] * 100)
            if results["gaps_before"] > 0
            else 0
        )

        logger.info("=" * 80)
        logger.info("🎉 GAP FIXING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Files processed: {results['files_processed']}")
        logger.info(f"📊 Gaps before: {results['gaps_before']}")
        logger.info(f"📊 Gaps after: {results['gaps_after']}")
        logger.info(f"📊 Gaps eliminated: {gaps_eliminated}")
        logger.info(f"📊 Improvement rate: {improvement_rate:.1f}%")
        logger.info(
            f"📊 Fix attempts: {results['gaps_fixed']} successful, {results['failed_fixes']} failed"
        )

        if results["errors"]:
            logger.warning(f"⚠️ {len(results['errors'])} errors occurred:")
            for error in results["errors"][:5]:  # Show first 5 errors
                logger.warning(f"   • {error}")
            if len(results["errors"]) > 5:
                logger.warning(f"   • ... and {len(results['errors']) - 5} more errors")

        results["improvement_rate"] = improvement_rate
        results["gaps_eliminated"] = gaps_eliminated

        return results

    except Exception as e:
        logger.error(f"❌ Critical error in gap fixing: {e}")
        results["errors"].append(str(e))
        return results


async def main():
    """Main function to fix gaps in problematic files"""

    # Target the files with the most gaps
    problematic_files=[
        "aggtrades_BINANCE_ETHUSDT_2023-09-24.parquet",  # 31 gaps
        "aggtrades_BINANCE_ETHUSDT_2023-08-27.parquet",  # 23 gaps
        "aggtrades_BINANCE_ETHUSDT_2023-09-23.parquet",  # 16 gaps
    ]

    logger.info("🎯 TARGETING MOST PROBLEMATIC FILES")
    logger.info(f"📋 Target files: {problematic_files}")
    logger.info("=" * 80)

    # Fix gaps in these specific files
    results=await fix_specific_files_gaps(
        symbol="ETHUSDT", exchange="BINANCE", target_files=problematic_files
    )

    # Success check
    if results["improvement_rate"] > 50:
        logger.info("🎉 SUCCESS: Significant improvement achieved!")
        return True
    elif results["improvement_rate"] > 0:
        logger.info("✅ PARTIAL SUCCESS: Some improvement achieved")
        return True
    else:
        logger.warning("⚠️ LIMITED SUCCESS: Minimal or no improvement")
        return False


if __name__== "__main__":
    success = asyncio.run(main())
    print(f"\nScript completed with {'success' if success else 'limited success'}")
    sys.exit(0 if success else 1)