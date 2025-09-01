#!/usr/bin/env python3
"""
Update Aggtrades Gaps - Comprehensive Gap Fixing Tool

This script provides multiple options for fixing gaps in aggtrades files:
    pass  # TODO: Add implementation
1. Fix gaps in specific files
2. Fix gaps in all files with >N gaps
3. Fix all gaps in the dataset

Usage:
    python update_aggtrades_gaps.py --mode specific --files file1.parquet file2.parquet
    python update_aggtrades_gaps.py --mode threshold --min-gaps 10
    python update_aggtrades_gaps.py --mode all
"""

import argparse
import asyncio
import sys
from pathlib import Path

from src.training.steps.step1.data_gap_detector import DataGapDetector
from src.training.steps.step1.missing_data_downloader_and_gap_filler import (
    MissingDataDownloaderAndGapFiller,
)
from src.utils.logger import system_logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("UpdateAggtradesGaps")

# Constants
EXCELLENT_IMPROVEMENT_THRESHOLD = 50


class AggtradesGapUpdater:
    """Comprehensive tool for updating and fixing aggtrades gaps"""

    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
        self.symbol = symbol
        self.exchange = exchange
        self.gap_detector = DataGapDetector()
        self.gap_filler = MissingDataDownloaderAndGapFiller()

    async def analyze_gaps(self) -> dict:
        """Analyze current gap situation"""
        logger.info("📊 ANALYZING CURRENT GAP SITUATION")
        logger.info("=" * 80)

        gaps = self.gap_detector.detect_aggtrades_gaps(self.symbol, self.exchange)

        # Group gaps by file
        gaps_by_file = {}
        for gap in gaps:
            file_name = gap["file"]
            if file_name not in gaps_by_file:
                gaps_by_file[file_name] = []
            gaps_by_file[file_name].append(gap)

        # Calculate statistics
        files_with_gaps = len(gaps_by_file)
        total_gaps = len(gaps)

        if gaps_by_file:
            gaps_per_file = [
                (file, len(file_gaps)) for file, file_gaps in gaps_by_file.items()
            ]
            gaps_per_file.sort(key=lambda x: x[1], reverse=True)

            logger.info("📊 FILES WITH MOST GAPS:")
            for i, (file_name, gap_count) in enumerate(gaps_per_file[:10], 1):
                logger.info(f"   {i:2d}. {file_name}: {gap_count} gaps")

        return {
            "total_gaps": total_gaps,
            "files_with_gaps": files_with_gaps,
            "gaps_by_file": gaps_by_file,
            "top_files": gaps_per_file[:20] if gaps_by_file else [],
        }

    async def fix_specific_files(self, target_files: list[str]) -> dict:
        """Fix gaps in specific files"""
        logger.info("🎯 FIXING GAPS IN SPECIFIC FILES")
        logger.info(f"📋 Target files: {target_files}")
        logger.info("=" * 80)

        # Get gaps for target files only
        all_gaps = self.gap_detector.detect_aggtrades_gaps(self.symbol, self.exchange)
        target_gaps = [gap for gap in all_gaps if gap["file"] in target_files]

        if not target_gaps:
            logger.info("✅ No gaps found in specified files")
            return {"success": True, "gaps_fixed": 0, "gaps_failed": 0}

        unique_files = {gap["file"] for gap in target_gaps}
        logger.info(
            f"🔧 Found {len(target_gaps)} gaps to fix in {len(unique_files)} files",
        )

        # Fix the gaps
        return await self.gap_filler.fill_aggtrades_gaps(
            self.symbol, self.exchange, target_gaps,
        )

    async def fix_files_with_min_gaps(self, min_gaps: int) -> dict:
        """Fix gaps in all files that have at least N gaps"""
        logger.info(f"🎯 FIXING GAPS IN FILES WITH ≥{min_gaps} GAPS")
        logger.info("=" * 80)

        analysis = await self.analyze_gaps()

        # Find files with enough gaps
        target_files = [
            file_name
            for file_name, gap_count in analysis["top_files"]
            if gap_count >= min_gaps
        ]

        if not target_files:
            logger.info(f"✅ No files found with ≥{min_gaps} gaps")
            return {"success": True, "gaps_fixed": 0, "gaps_failed": 0}

        logger.info(f"🔧 Will fix gaps in {len(target_files)} files:")
        for file_name in target_files:
            gap_count = analysis["gaps_by_file"][file_name]
            logger.info(f"   • {file_name}: {len(gap_count)} gaps")

        return await self.fix_specific_files(target_files)

    async def fix_all_gaps(self) -> dict:
        """Fix all gaps in the dataset"""
        logger.info("🌟 FIXING ALL GAPS IN DATASET")
        logger.info("=" * 80)

        all_gaps = self.gap_detector.detect_aggtrades_gaps(self.symbol, self.exchange)

        if not all_gaps:
            logger.info("✅ No gaps found in dataset")
            return {"success": True, "gaps_fixed": 0, "gaps_failed": 0}

        logger.info(f"🔧 Found {len(all_gaps)} total gaps to fix")

        # Fix all gaps
        return await self.gap_filler.fill_aggtrades_gaps(
            self.symbol, self.exchange, all_gaps,
        )

    async def verify_improvements(self, before_analysis: dict) -> dict:
        """Verify gap fixing improvements"""
        logger.info("📊 VERIFYING IMPROVEMENTS")
        logger.info("=" * 80)

        after_analysis = await self.analyze_gaps()

        gaps_before = before_analysis["total_gaps"]
        gaps_after = after_analysis["total_gaps"]
        gaps_eliminated = gaps_before - gaps_after
        improvement_rate = (
            (gaps_eliminated / gaps_before * 100) if gaps_before > 0 else 0
        )

        logger.info("📊 IMPROVEMENT SUMMARY:")
        logger.info(f"   • Gaps before: {gaps_before}")
        logger.info(f"   • Gaps after: {gaps_after}")
        logger.info(f"   • Gaps eliminated: {gaps_eliminated}")
        logger.info(f"   • Improvement rate: {improvement_rate:.1f}%")

        return {
            "gaps_before": gaps_before,
            "gaps_after": gaps_after,
            "gaps_eliminated": gaps_eliminated, "improvement_rate": improvement_rate,
        }


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Update and fix aggtrades gaps")
    parser.add_argument(
        "--mode",
        choices=["specific", "threshold", "all", "analyze"],
        default="analyze",
        help="Mode of operation",
    )
    parser.add_argument(
        "--files", nargs="+", help="Specific files to fix (for specific mode)",
    )
    parser.add_argument(
        "--min-gaps",
        type=int,
        default=10,
        help="Minimum gaps threshold (for threshold mode)",
    )
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name")

    args = parser.parse_args()

    # Initialize updater
    updater = AggtradesGapUpdater(args.symbol, args.exchange)

    # Get initial analysis
    logger.info("🚀 STARTING GAP UPDATE PROCESS")
    before_analysis = await updater.analyze_gaps()

    if args.mode == "analyze":
        logger.info("✅ Analysis complete. Use --mode to perform fixes.")
        return True

    # Perform the requested operation
    if args.mode == "specific":
        if not args.files:
            logger.error("❌ --files required for specific mode")
            return False
        await updater.fix_specific_files(args.files)

    elif args.mode == "threshold":
        await updater.fix_files_with_min_gaps(args.min_gaps)

    elif args.mode == "all":
        await updater.fix_all_gaps()

    # Verify improvements
    improvements = await updater.verify_improvements(before_analysis)

    # Final summary
    logger.info("=" * 80)
    if improvements["improvement_rate"] > EXCELLENT_IMPROVEMENT_THRESHOLD:
        logger.info("🎉 EXCELLENT: Significant improvement achieved!")
        success = True
    elif improvements["improvement_rate"] > 0:
        logger.info("✅ GOOD: Some improvement achieved")
        success = True
    else:
        logger.warning("⚠️ LIMITED: Minimal or no improvement")
        success = False

    logger.info("=" * 80)
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\nScript completed with {'success' if success else 'limited success'}")
    sys.exit(0 if success else 1)
